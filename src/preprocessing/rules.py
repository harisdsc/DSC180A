import re

class TransactionCleaner:
    def __init__(self):
        self.known_merchants = [
            # specific cases first
            (re.compile(r"AMZN\s?MKTP|AMZN\s+COM\s+BILL|AMAZON(?:\.COM|\s*MKTPLACE|\s*PRIME)?"), "AMAZON"), 
            (re.compile(r"AMZN"), "AMAZON"),
            (re.compile(r"(?:THE\s+)?HOME\s+DEPOT"), "THE HOME DEPOT"),
            (re.compile(r"WAL-?MART"), "WALMART"),
            (re.compile(r"TARGET"), "TARGET"),
            (re.compile(r"UBER(?:\s*EATS|\s*TRIP)?"), "UBER"),
            (re.compile(r"LYFT"), "LYFT"),
            (re.compile(r"DOORDASH"), "DOORDASH"),
            (re.compile(r"NETFLIX"), "NETFLIX"),
            (re.compile(r"STARBUCKS"), "STARBUCKS"),
            (re.compile(r"MCDONALD'?S"), "MCDONALDS"),
            (re.compile(r"7-?ELEVEN"), "7-ELEVEN"),
            (re.compile(r"CHICK-?FIL-?A"), "CHICK-FIL-A"),
            (re.compile(r"DUNKIN"), "DUNKIN"),
            (re.compile(r"GODADDY"), "GODADDY"),
            (re.compile(r"CASH\s+APP"), "CASH APP"),
        ]
        
        self.cleaning_steps = [
            # === A) Masking/Patterns ===
            (re.compile(r"\b[A-Z]{2}\s+[SP]X{8,}(?:\s+CARD)?(?:\s+X+)?\b"), " "), 
            (re.compile(r"\b[SP]X{8,}(?:\s+CARD)?(?:\s+X+)?\b"), " "),
            (re.compile(r"\bCOM\s+BILL\s*(?:[A-Z]{2})?\b"), " "),

            # === B) Standard Cleaning ===
            # Headers
            (re.compile(r"^(?:RECURRING\s+PAYMENT|(?:RECURRING|POS|DEBIT)\s+)?\s*(?:PURCHASE|PYMT|PAYMENT|TRANS|TRANSACTION)?\s+(?:AUTHORIZED|WITHDRAWAL)\s*(?:ON)?"), " "),
            
            # Card boilerplate
            (re.compile(r"\b(?:VISA|MC|MASTERCARD|AMEX|DISCOVER)\s+(?:CHECK\s+)?(?:CARD|PURCHASE|PAYMENT|DEBIT)\b"), " "),
            (re.compile(r"\b(?:POS|DEBIT)\s+(?:DEBIT|PURCHASE)\b"), " "),
            (re.compile(r"\b(?:CHECK)?\s*CARD\s+(?:#|X+|ENDING IN|\d{4})"), " "), # Upper case check
            (re.compile(r"\bCARD\s+X+\b"), " "),

            # Codes/Dates/Phones
            (re.compile(r"\b[A-Z0-9]{2,}\*\*\*\*\*+[A-Z0-9]{4}\b"), " "),
            (re.compile(r"\b(?:S|L)?\d{5,}\b"), " "),
            (re.compile(r"\bX{3,}\b"), " "),
            (re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"), " "),
            (re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"), " "),
            
            # Trailing garbage
            (re.compile(r"\b(ID|REF|SEQ|CODE|AUTH)\s*#?[:\s]*\w+\b"), " "),
        ]

        noise_words = [
            "DBT", "PURCH", "TRANSACTION", "HTTPS", "WWW", "CONSUMER", "CKCD",
            "CRD", "PUR", "LLC", "INC", "SIGNATURE", "WEB", "PAYMENT", "DEB",
            "INTL", "RECURRING", "DIGIT", "ONLINE", "WITHDRAWAL", "RESTAURANT",
            "E-COMMERCE", "BUSINESS", "POS", "PURCHASE", "DEBIT", "HELP",
            "USA", "US", "TERMINAL", "CHECKCARD"
        ]
        # Ensure boundaries are strictly enforced
        self.noise_regex = re.compile(r"\b(?:" + "|".join(noise_words) + r")\b")

        city_list = [
            "MIAMI", "PHOENIX", "SEATTLE", "HOUSTON", "ORLANDO", "CHICAGO",
            "ATLANTA", "LAS VEGAS", "CHARLOTTE", "TAMPA", "GREENVILLE", "BROOKLYN",
            "DENVER", "LOS ANGELES", "SAN ANTONIO", "MEMPHIS", "NEW YORK",
            "RICHMOND", "INDIANAPOLIS", "COLUMBUS", "PHILADELPHIA", "AUSTIN",
            "SAN FRANCISCO", "SAN DIEGO", "DALLAS", "BOSTON"
        ]
        self.city_regex = re.compile(r"\b(?:" + "|".join(city_list) + r")\b")

        state_list = [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
            "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME",
            "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM",
            "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
            "UT", "VA", "VT", "WA", "WI", "WV", "WY"
        ]
        self.state_end_regex = re.compile(r"\b(?:" + "|".join(state_list) + r")\s*$")

        # Cleanup helpers
        self.space_normalizer = re.compile(r"[\u00A0\s]+")
        self.punctuation_cleaner = re.compile(r"[^\w\s'&]")
        self.trailing_com = re.compile(r"\.COM$")

    def clean(self, raw_memo):
        if not isinstance(raw_memo, str): return ""

        # 1. Global Prep
        memo = raw_memo.upper().strip()

        # 2. Fast Short Circuit
        for regex, clean_name in self.known_merchants:
            if regex.search(memo):
                return clean_name

        # 3. Apply Regex Pipeline (Ordered strictly to avoid 'while' loop)
        
        # A. Remove Structural Garbage (Masks, Dates, Codes)
        for regex, replacement in self.cleaning_steps:
            memo = regex.sub(replacement, memo)

        # B. Remove Noise Words
        memo = self.noise_regex.sub(" ", memo)

        # C. Remove Cities (Do this before punctuation to catch "Miami," etc if needed, 
        # though regex handles boundaries)
        memo = self.city_regex.sub(" ", memo)

        # D. Normalize Punctuation (Chars -> Space)
        memo = self.punctuation_cleaner.sub(" ", memo)
        
        # E. Normalize Spaces (Collapse multiple spaces to one)
        memo = self.space_normalizer.sub(" ", memo).strip()

        # F. Remove State at End (Must be done after stripping spaces)
        memo = self.state_end_regex.sub("", memo).strip()

        # G. Final Polish (Trailing .COM, etc)
        memo = self.trailing_com.sub("", memo)

        return memo