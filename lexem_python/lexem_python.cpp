#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <set>
using namespace std;


const string CONFIG_PREFIX = R"(C:\Users\bidzi\Documents\sheva\sp\lab3\config\)";
const bool USE_COLOR = true;

enum LexemType {
    NUM = 0,
    HEX,
    CNST,
    DIRECTIVE,
    KEYWORD,
    OPERATOR,
    PUNCTUATION,
    IDENTIFIER,
    SPACES,
    COMMENT,
    FUNCTION,
    UNKNOWN
};

string LexemTypeNames[] = { 
    "NUM",
    "HEX",
    "CNST",
    "DIRECTIVE",
    "KEYWORD",
    "OPERATOR",
    "PUNCTUATION",
    "IDENTIFIER",
    "SPACES",
    "COMMENT",
    "FUNCTION",
    "UNKNOWN" 
};

namespace Color {
    enum Code {
        BG_GREY = 0,
        BG_RED = 1,
        BG_GREEN = 2,
        BG_YELLOW = 3,
        BG_BLUE = 4,
        BG_VIOLET = 5,
        BG_DARK_VIOLET = 13,
        BG_CYAN = 6,
        BG_WHITE = 15,
        BG_LIGHT_GREEN = 10,
        BG_DARK_RED = 52,
    };
    class Modifier {
        Code code;
    public:
        Modifier(Code pCode) : code(pCode) {}
        friend std::ostream&
            operator<<(std::ostream& os, const Modifier& mod) {
            return os << "\033[48;5;" << mod.code << "m";
        }
    };
}
Color::Modifier LexemColors[] = {
    Color::BG_CYAN,
    Color::BG_DARK_RED,
    Color::BG_VIOLET,
    Color::BG_YELLOW,
    Color::BG_BLUE,
    Color::BG_DARK_VIOLET,
    Color::BG_GREY,
    Color::BG_GREEN,
    Color::BG_GREY,
    Color::BG_LIGHT_GREEN,
    Color::BG_YELLOW,
    Color::BG_RED
};

struct Lexem {
    LexemType type;
    string content;
    Lexem(LexemType type, string content) : type(type), content(content) {};
};

ostream& operator<<(ostream& os, const Lexem& lexem)
{
    if (USE_COLOR)
    {
        if (lexem.content == "\n") os << Color::Modifier(Color::BG_GREY) << Color::Modifier(Color::BG_GREY);
        os << LexemColors[(int)lexem.type] << lexem.content << Color::Modifier(Color::BG_GREY);
    }
    else 
        os << LexemTypeNames[(int)lexem.type] << " \"" << lexem.content << "\"" << '\n';
    
    return os;
}

set<string> punctiations;
set<string> operators; 
set<string> operator_prefixes;
set<string> keywords;
set<string> identifier_starters;
set<string> identifier_suffixes;

bool is_space(string s) {
    if (s.size() != 1)return false;
    char c = s[0];
    return c == ' ' || c == '\n' || c == '\t' || c == '\r';
}

bool is_punct(string s) {
    return punctiations.count(s) > 0;
}

bool is_operator(string s) {
    return operators.count(s) > 0;
}
bool is_operator_prefix(string s) {
    return operator_prefixes.count(s) > 0;
}

bool is_character(string s) {
    return ('a' <= s[0] && s[0] <= 'z') || ('A' <= s[0] && s[0] <= 'Z');
}
bool is_digit(string s) {
    return '0' <= s[0] && s[0] <= '9';
}
bool is_keyword(string s) {
    return keywords.count(s) > 0;
}
bool is_identifier_start(string s) {
    return identifier_starters.count(s) > 0;
}
bool is_identifier_suffix(string s) {
    return identifier_suffixes.count(s) > 0;
}

vector<Lexem> parse(const string filename) {
    ifstream f(filename);
    vector<Lexem> ret;
    string current = "";
    auto add = [&](LexemType type) {ret.push_back(Lexem(type, current)); current.clear(); };
    while (f.peek() != EOF){
        current.push_back(f.get());
        if (is_space(current)) add(LexemType::SPACES);
        else if (is_punct(current)) 
            add(LexemType::PUNCTUATION);
        else if (current == ".") {
            Lexem prev = ret.back();
            if (prev.type == LexemType::OPERATOR || prev.type == LexemType::IDENTIFIER || prev.type == LexemType::CNST) {
                add(LexemType::IDENTIFIER);
            }
            else {
                add(LexemType::UNKNOWN);
            }
        }
        else if (current == "\"" || current == "'") {
            while (!f.eof()) {
                char c = f.get();
                current.push_back(c);
                if (c == current[0]) break;
            }
            add(LexemType::CNST);
        }
        else if (is_operator(current)) {
            while (!f.eof() && is_operator(string(1, f.peek()))) {
                current.push_back(f.peek());
                if (!is_operator(current) && !is_operator_prefix(current)) {
                    current.pop_back();
                    break;
                }
                else {
                    f.get();
                }
            }
            if (!is_operator(current)) {
                add(LexemType::UNKNOWN);
            }
            else {
                if (current == "(") {
                    if (ret.back().type == LexemType::IDENTIFIER)
                        ret.back().type = LexemType::FUNCTION;
                }
                add(LexemType::OPERATOR);
            }
        }
        else if (is_identifier_start(current)) {
            bool was_dot = false; 
            while ( (was_dot == false && (is_identifier_suffix(string(1, f.peek())) || f.peek() == '.')) ||
                (was_dot == true && is_identifier_start(string(1, f.peek())))) {
                if (f.peek() == '.') was_dot = true;
                current.push_back(f.get());
            }
            if (is_keyword(current)) {
                add(LexemType::KEYWORD);
            }
            else {
                add(LexemType::IDENTIFIER);
            }
        }
        else if (is_digit(current)) {
            while (is_digit(string(1, f.peek())) || f.peek() == '.') {
                current.push_back(f.get());
            }
            if (is_digit(current)) add(LexemType::CNST);
            else add(LexemType::UNKNOWN);
        }
        else if (current == "#") {
            while (!f.eof() && f.peek() != '\n') {
                current.push_back(f.get());
            }
            add(LexemType::COMMENT);
           
        }
        else {
            cout << "WARN: SHOULD NOT HAPPEN!\n";
            add(LexemType::UNKNOWN);
        }
    }

    f.close();
    return ret;

}

void init() {
    ifstream f(CONFIG_PREFIX + "keywords.txt");
    string s;
    while (f >> s) {
        keywords.insert(s);
    }
    f.close();
    f.open(CONFIG_PREFIX + "operators.txt");
    while (f >> s) {
        operators.insert(s);
        for (int i = 1; i < s.size(); i++)
            operator_prefixes.insert(s.substr(0, i));
    }
    f.close();
    f.open(CONFIG_PREFIX + "punct.txt");
    while (f >> s) {
        punctiations.insert(s);
    }
    f.close();
    f.open(CONFIG_PREFIX + "identifier_start.txt");
    while (f >> s) {
        identifier_starters.insert(s);
    }
    f.close();
    f.open(CONFIG_PREFIX + "identifier_suffix.txt");
    while (f >> s) {
        identifier_suffixes.insert(s);
    }
    f.close();
}

int main()
{
    init();
    cout << "Please input filename for parsing:\n>>> ";
    string filename; cin >> filename;
    auto res = parse(filename);
    for (auto x : res) {
        cout << x;
    }
}