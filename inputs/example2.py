from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,ExtraTreesClassifier
import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle as pk
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

class EncouragingLoss(nn.Module):
    def __init__(self, log_end=0.75, reduction='mean'):
        super(EncouragingLoss, self).__init__()
        self.log_end = log_end  # 1 refers to the normal bonus, but 0.75 can easily work in existing optimization systems, 0.5 work for all settings we tested, recommend LE=0.75 for high accuracy scenarios and low LE for low accuracy scenarios.
        self.reduction = reduction

    def forward(self, input, target):
        lprobs = F.log_softmax(input, dim = -1)  # logp
        probs = torch.exp(lprobs)
        bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=1e-5))  # log(1-p)
        if self.log_end != 1.0:  # end of the log curve in conservative bonus 
            log_end = self.log_end
            y_log_end = torch.log(torch.ones_like(probs) - log_end)
            bonus_after_log_end = 1/(log_end - torch.ones_like(probs)) * (probs-log_end) + y_log_end
            bonus = torch.where(probs > log_end, bonus_after_log_end, bonus)
        loss = F.nll_loss(lprobs-bonus, target.view(-1), reduction=self.reduction)
        return loss

class LightningModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = EncouragingLoss()
        self.model = nn.Sequential(nn.Linear(21, 50), nn.ReLU(), nn.Linear(50, 40), nn.ReLU(), nn.Linear(40, 15), nn.ReLU(), nn.Linear(15, 2))
    def forward(self, X):
        return self.model(X)
    def training_step(self, batch, batch_id):
        X, y = batch
        X = X.view(X.size(0), -1)
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_id):
        X, y = batch
        X = X.view(X.size(0), -1)
        y_hat = self.model(X)
        preds = F.softmax(y_hat, dim = 1)[:, -1]
        score = roc_auc_score(np.array(y.cpu()).reshape(-1), np.array(preds.cpu()).reshape(-1))
        self.log("val_auc", score, prog_bar=True, on_step=False, on_epoch=True)
        return score
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        

class AlternativeLightningModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = EncouragingLoss()
        self.model = nn.Sequential(nn.BatchNorm1d(21), nn.Linear(21, 30), nn.ReLU(), nn.Linear(30, 100), nn.ReLU(), nn.Linear(100, 1))
    def forward(self, X):
        return self.model(X)
    def training_step(self, batch, batch_id):
        X, y = batch
        X = X.view(X.size(0), -1)
        y_hat = F.sigmoid(self.model(X))
        y_hat = torch.stack([1 - y_hat, y_hat], dim = 1).squeeze()
        y_hat = torch.log(y_hat / (1 - y_hat))
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_id):
        X, y = batch
        X = X.view(X.size(0), -1)
        y_hat = self.model(X)
        preds = F.sigmoid(y_hat)
        score = roc_auc_score(np.array(y.cpu()).reshape(-1), np.array(preds.cpu()).reshape(-1))
        self.log("val_auc", score, prog_bar=True, on_step=False, on_epoch=True)
        return score
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

from sklearn.base import BaseEstimator, ClassifierMixin
class ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    
    def predict_proba(self, X):
        with torch.no_grad():
            X = torch.tensor(np.array(X), dtype = torch.float)
            X = X.view(X.size(0), -1)
            y = self.model(X)
            if (y.shape[-1] == 2):
                y = F.softmax(y, dim = -1)[:, -1].cpu().numpy()
            else:
                y = F.sigmoid(y).cpu().squeeze().numpy()
            return y
    
    def fit(self, X, y):
        trainer = L.Trainer(max_epochs=100)
        X_dataset = data.TensorDataset(torch.tensor(X.to_numpy(), dtype = torch.float), torch.tensor(y.to_numpy()))
        trainer.fit(self.model, data.DataLoader(X_dataset, batch_size = 512, num_workers = 2))
    

class BoosterModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, params):
        super().__init__()
        self.model = None
        self.nb = params.pop('num_round')
        self.params = params 
    def fit(self, X, y):
        train = lgb.Dataset(X, y)
        self.model = lgb.train(self.params, num_boost_round=self.nb, train_set=train)
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    def predict_proba(self, X):
        return self.model.predict(X)


class ClippingRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self):
        def frwrd(x):
            y = 0.3 + x * (1 - 2 * 0.3)
            return np.log(y/(1-y))
        def invrs(x):
            y = 1/(1 + np.exp(-x))
            return (y - 0.3) / (1 - 2 * 0.3)

        self.base_regressor = make_pipeline(
            ColumnTransformer([('keep', 'passthrough', ['l', 'b', 'locCodeAndComment', 'branchCount', 'v(g)', 'lOBlank', 'ev(g)', 'd', 'uniq_Op', 'uniq_Opnd', 'loc', 'lOCode'])]),
            TransformedTargetRegressor(RidgeCV(), func=frwrd, inverse_func=invrs)
        )

        self.classes_ = [0, 1]

    def fit(self, X, y):
        self.base_regressor.fit(X, y)
        return self

    def predict(self, X):
        # Predict using the base model
        predictions = self.base_regressor.predict(X)

        # Clip the predictions to the [0, 1] range
        clipped_predictions = np.clip(predictions, 0, 1)
        
        return clipped_predictions

    def predict_proba(self, X):
        return self.predict(X)
    
class MODELS:

    ensemble_1 = VotingClassifier(
        estimators=[RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), HistGradientBoostingClassifier(), ExtraTreesClassifier()],
        weights=[0.1897294849783498, 0.1935407721546804, 0.9507737143576438, 0.7603155650474646, 0.023127372301846052])

    ensemble_2 = VotingClassifier(
        estimators=[ModelWrapper(LightningModel()), 
        ModelWrapper(AlternativeLightningModel()), 
        AdaBoostClassifier(n_estimators= 17, base_estimator = DecisionTreeClassifier(criterion= "log_loss", max_depth= 5)),
        RandomForestClassifier(**{"n_estimators": 171, "criterion": "log_loss", "max_depth": 8, "max_features": 0.5}),
        BoosterModelWrapper({
            'num_leaves': 20,
            'num_round': 66,
            'objective': 'cross_entropy',
            'learning_rate': 0.05783733512308553})
        ], 
        weights=[0.8363550163413104, 0.05726822194323267, 0.7458115338767694, 0.6651879688596302, 0.5157172196665011]
    )

    dnn_1 = LightningModel()
    #dnn_1.load_state_dict(torch.load(r'C:\Users\bidzi\Documents\ml\kaggle_playground\10.23\models-checkpoints\DNN-arhictecture-1.pt'))
    dnn_1 = ModelWrapper(dnn_1)

    dnn_2 = AlternativeLightningModel()
    #dnn_2.load_state_dict(torch.load(r'C:\Users\bidzi\Documents\ml\kaggle_playground\10.23\models-checkpoints\DNN-arhictecture-2.pt'))
    dnn_2 = ModelWrapper(dnn_2)

    clipped_ridge = ClippingRegressor()

