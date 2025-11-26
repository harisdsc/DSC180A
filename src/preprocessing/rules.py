import re

class TransactionCleaner:
    def __init__(self):
        # 1. High Confidence Merchants (Short-Circuit)
        self.known_merchants = [
            (re.compile(r".*(AMAZON(?:\.COM|\s*MKTPLACE|\s*PRIME)?).*", re.I), "AMAZON"),
            (re.compile(r".*(AMZN\s?MKTP|AMZN\s+COM\s+BILL).*", re.I), "AMAZON"),
            (re.compile(r".*(AMZN).*", re.I), "AMAZON"),  # Fallback for just "AMZN"
            (re.compile(r".*((?:THE\s+)?HOME\s+DEPOT).*", re.I), "THE HOME DEPOT"),
            (re.compile(r".*(WAL-?MART).*", re.I), "WALMART"),
            (re.compile(r".*(TARGET).*", re.I), "TARGET"),
            (re.compile(r".*(UBER(?:\s*EATS|\s*TRIP)?).*", re.I), "UBER"),
            (re.compile(r".*(LYFT).*", re.I), "LYFT"),
            (re.compile(r".*(DOORDASH).*", re.I), "DOORDASH"),
            (re.compile(r".*(NETFLIX).*", re.I), "NETFLIX"),
            (re.compile(r".*(STARBUCKS).*", re.I), "STARBUCKS"),
            (re.compile(r".*(MCDONALD'?S).*", re.I), "MCDONALDS"),
            (re.compile(r".*(7-?ELEVEN).*", re.I), "7-ELEVEN"),
            (re.compile(r".*(CHICK-?FIL-?A).*", re.I), "CHICK-FIL-A"),
            (re.compile(r".*(DUNKIN).*", re.I), "DUNKIN"),
            (re.compile(r".*(GODADDY).*", re.I), "GODADDY"),
            (re.compile(r".*(CASH\s+APP).*", re.I), "CASH APP"),
        ]

        # 2. General Cleaning Rules (Executed if no known merchant found)
        self.cleaning_steps = [
            # === A) Massive N-Gram Noise (New Rules) ===

            # 1. The "State + S/P Mask" Pattern (e.g., "CA SXXXXXXXXXXXXXXX CARD")
            # We remove the State + Mask + Optional Card suffix
            (re.compile(r"\b[A-Z]{2}\s+[SP]X{8,}(?:\s+CARD)?(?:\s+X+)?\b", re.I), " "),

            # 2. The Standalone S/P Mask (e.g., "SXXXXXXXXXXXXXXX")
            (re.compile(r"\b[SP]X{8,}(?:\s+CARD)?(?:\s+X+)?\b", re.I), " "),

            # 3. "COM BILL" Pattern (e.g., "COM BILL WA")
            # Often appears at the end of online transactions
            (re.compile(r"\bCOM\s+BILL\s*(?:[A-Z]{2})?\b", re.I), " "),

            # === B) Standard Cleaning ===

            # Normalize spaces
            (re.compile(r"[\u00A0\s]+"), " "),

            # Headers/Prefixes (Updated with "RECURRING PAYMENT")
            (re.compile(
                r"^(?:RECURRING\s+PAYMENT|(?:RECURRING|POS|DEBIT)\s+)?\s*(?:PURCHASE|PYMT|PAYMENT|TRANS|TRANSACTION)?\s+(?:AUTHORIZED|WITHDRAWAL)\s*(?:ON)?",
                re.I), " "),

            # Card boilerplate (Updated for "VISA CHECK CARD", "POS DEBIT")
            (re.compile(r"\b(?:VISA|MC|MASTERCARD|AMEX|DISCOVER)\s+(?:CHECK\s+)?(?:CARD|PURCHASE|PAYMENT|DEBIT)\b",
                        re.I), " "),
            (re.compile(r"\b(?:POS|DEBIT)\s+(?:DEBIT|PURCHASE)\b", re.I), " "),  # Handles "POS DEBIT", "DEBIT PURCHASE"
            (re.compile(r"\b(?:CHECK)?\s*CARD\s+(?:#|X+|Ending In|\d{4})", re.I), " "),
            (re.compile(r"\bCARD\s+X+\b", re.I), " "),  # Explicit "CARD XXXX"

            # Ugly Numbers/Codes
            (re.compile(r"\b[A-Z0-9]{2,}\*\*\*\*\*+[A-Z0-9]{4}\b"), " "),
            (re.compile(r"\b(?:S|L)?\d{5,}\b"), " "),
            (re.compile(r"\bX{3,}\b", re.I), " "),

            # Dates/Phones
            (re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?", re.I), " "),
            (re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"), " "),

            # Garbage Tails
            (re.compile(r"\b(ID|REF|SEQ|CODE|AUTH)\s*#?[:\s]*\w+\b", re.I), " "),
        ]

        # 3. Noise Words (Updated from N-Grams)
        noise_words = [
            "DBT", "PURCH", "TRANSACTION", "HTTPS", "WWW", "CONSUMER", "CKCD",
            "CRD", "PUR", "LLC", "INC", "SIGNATURE", "WEB", "PAYMENT", "DEB",
            "INTL", "RECURRING", "DIGIT", "ONLINE", "WITHDRAWAL", "RESTAURANT",
            "E-COMMERCE", "BUSINESS", "POS", "PURCHASE", "DEBIT", "HELP",
            "USA", "US", "TERMINAL", "CHECKCARD"
        ]
        self.noise_regex = re.compile(r"\b(" + "|".join(noise_words) + r")\b", re.IGNORECASE)

        # 4. Geography (Cities/States)
        city_list = [
            "MIAMI", "PHOENIX", "SEATTLE", "HOUSTON", "ORLANDO", "CHICAGO",
            "ATLANTA", "LAS VEGAS", "CHARLOTTE", "TAMPA", "GREENVILLE", "BROOKLYN",
            "DENVER", "LOS ANGELES", "SAN ANTONIO", "MEMPHIS", "NEW YORK",
            "RICHMOND", "INDIANAPOLIS", "COLUMBUS", "PHILADELPHIA", "AUSTIN",
            "SAN FRANCISCO", "SAN DIEGO", "DALLAS", "BOSTON"
        ]
        self.city_regex = re.compile(r"\b(" + "|".join(city_list) + r")\b", re.IGNORECASE)

        state_list = [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
            "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME",
            "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM",
            "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
            "UT", "VA", "VT", "WA", "WI", "WV", "WY"
        ]
        # Matches State only if it is at the END of the string
        self.state_end_regex = re.compile(r"\b(" + "|".join(state_list) + r")\s*$", re.IGNORECASE)

    def clean(self, raw_memo):
        if not isinstance(raw_memo, str): return ""

        # 1. Basic Prep
        memo = raw_memo.upper().strip()

        # 2. Short Circuit (Known Merchants)
        for regex, clean_name in self.known_merchants:
            if regex.search(memo):
                return clean_name

        # 3. General Cleaning
        for regex, replacement in self.cleaning_steps:
            memo = regex.sub(replacement, memo)

        # 4. Noise Removal
        memo = self.noise_regex.sub(" ", memo)

        # 5. Iterative Cleanup (Geography & Tails)
        prev_memo = None
        while memo != prev_memo:
            prev_memo = memo

            # Remove City names
            memo = self.city_regex.sub(" ", memo)

            # Remove State ONLY at the end
            memo = self.state_end_regex.sub("", memo)

            # Normalize punctuation
            memo = re.sub(r"[^\w\s'&]", " ", memo)

            # Collapse spaces
            memo = re.sub(r"\s+", " ", memo).strip()

            # Remove trailing .COM
            memo = re.sub(r"\.COM$", "", memo)

        return memo
