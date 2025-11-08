import re

STATE_LIST = [ 
    "AL", "AK", "AZ", "AR", "CA", 
    "(?<!\.)CO(?!['`])", # Negative lookbehind/ahead for CO (e.g., not .CO or COSTCO) 
    "CT", "DC", "DE", "FL", "GA", "HI", "IA", 
    "ID", "IL", 
    "IN(?!\\s+N\\s+OUT\\s+BURGER)", # Negative lookahead for IN (not IN N OUT BURGER) 
    "KS", "KY", 
    "(?<!['`])LA(?!\\s+HACIENDA|\\s+FITNESS|\\s+LA'S|['`])", # Negative lookaheads for LA 
    "MA", "MD", 
    "ME(?!\\s+DIA)", # Negative lookahead for ME (not ME DIA) 
    "MI", "MN", "MO(?!['`])", 
    "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", 
    "OH", "OK", "OR", 
    "PA(?!['`])", 
    "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", 
    "WI", "WV", "WY" 
] 
STATE_REGEX = r"\b(" + "|".join(STATE_LIST) + r")\b"

CITY_LIST = [
    "MIAMI", "PHOENIX", "SEATTLE", "HOUSTON", "ORLANDO", 
    "CHICAGO", "ATLANTA", "LAS VEGAS", "CHARLOTTE", "TAMPA", 
    "GREENVILLE", "BROOKLYN", "DENVER", "LOS ANGELES", "SAN ANTONIO", "MEMPHIS", "NEW YORK",
    "RICHMOND", "MONT BELVIEU", "INDIANAPOLIS", "PINECREST", "COLUMBUS", "PHILADELPHIA"
    # Added from 2-grams
    "WAXAHACHIE", "EL CAJON", "PASO ROBLES", "BUENA VI", "CHULA VISTA", 
    "BOCA RATON", "PINE PLAINS", "HIGHLANDS RAN", "RENTON", "SALT", "BOGOT",
    "SAN FRANCISCO", "SAN DIEGO"
]
CITY_REGEX = r"\b(" + "|".join(CITY_LIST) + r")\b"

NOISE_WORDS = [
    "DBT", "PURCH", "TRANSACTION", "HTTPSWWW", "WWW", "CONSUMER", 
    "CKCD", "CRD", "PUR", "LLC", "SIGNATURE", "WEB", "PAYMENT",
    "DEB", "INTL", "RECURRING", "DIGIT", "ONLINE",
    "800", "888", "WITHDRAWAL", "STORE", "STORES", "RESTAURANT",
    "E-COMMERCE", "NEW", "PORT",
    "STOP", "BUSINESS",
    # Added from 1-gram
    "POS", "PURCHASE", "DEBIT",
    # Added from 2-grams
    "HELP", "HTTPSHBOMAX", "HTTPSINSTACAR"
]
NOISE_WORDS_REGEX = r"\b(" + "|".join(NOISE_WORDS) + r")\b"

REGEX_PRE = [ 
    # === 0) Normalize spaces first === 
    (r"\u00A0", " "), # Replace non-breaking space with regular space 
    (r"\s{2,}", " "), # Collapse multiple spaces into one 

    # === 1) “Authorized / Recurring” headers === 
    (r"\b(?:RECURRING\s+)?PAYMENT\s+AUTHORIZED\s+ON(?:\s+\d{2}[/-]\d{2,4})?\b", " "), 
    (r"\b(?:P?URCHASE\s+)?AUTHORIZED\s+ON(?:\s+\d{2}[/-]\d{2,4})?\b", " "), 
    (r"\bAUTHORIZED\s+ON\s+\d{2}[/-]\d{2,4}\b", " "), 
    (r"\bRECURRING\s+PYMT\b", " "), 

    # === 2) Card & mask boilerplate === 
    (r"\b(?:VISA|MASTERCARD|AMEX|DISCOVER)\s+CHECK\s+CARD\b", " "), 
    (r"\bCHECK\s*CARD\b(?:\s*X+)?", " "), 
    (r"\bCARD(?:\s+ENDING\s+IN)?\s*X{4}\b", " "), 
    (r"\bDEBIT\s+CARD\s+DEBIT\s*/", " "), 
    (r"\b(?:DEBIT|CREDIT)\s+CARD\s+(?:PURCHASE|DEBIT|AUTH(?:ORIZATION)?)\b", " "), 
    (r"\b(?:WITHDRAWAL|POS)\s*#", " "), 
    (r"\bWITHDRAWAL\s+DEBIT\s+CHIP\b", " "), 
    (r"\bPOS\s+PUR-\s*(?:\*+)?", " "), 
    (r"\bAUTH\s*#\s*-?", " "), 
    (r"\bCK\s*X+\b", " "), 
    (r"\bPOS\s+(?:PURCHASE|WITHDRAWAL|DEBIT)\b", " "), 
    (r"\b(?:DDA\s+)?PIN\s+POS\s+PUR\b", " "), 
    (r"\bCDX{4,}\b", " "), 
    (r"\bF[XF]{4,}\b", " "), # Handle FXXXXX, FXXXX 
    (r"\bXXX\b", " "), # Handle masked number from n-grams
    
    # Generalized from [SP]X{6,} to [A-Z]X{5,} to match SXXXXX
    # Moved *before* the generic X{4,} rule to ensure correct match priority
    (r"\b[A-Z]X{5,}\b", " "), 
    
    (r"X{4,}", " "), # Remove generic masked numbers 
    
    # (The old \b[SP]X{6,}\b rule is removed from its original position)
    
    (r"\bDEBIT\s+(?:CARD|CRD)\b", " "), 
    (r"\bDEBIT\s+PURCHASE\b", " "), 
    (r"\bPOS\s+SIGNATURE\b", " "), 
    (r"\b(?:VISA|MASTERCARD|AMEX|DISCOVER|CARD|DATE|MCC)\b", " "), # Remove common card-related keywords 
    (r"^\s*PURCHASE\b", " "), # Remove "PURCHASE" if at start 
    (r"^\s*REC\s+POS\b", " "), 
    (r"^\s*RECURRING\b", " "), 

    # === 2.5) Prefix Normalization === 
    (r"\b(DNH)(?=[A-Z]{2,})", r"\1 "), # Fix "DNHGODADDYCOM" -> "DNH GODADDYCOM" 
    (r"\bDD\s*(?:[\\/]\s*)?BR\b", "DDBR"), # Combine DD/BR or DD BR -> DDBR 

    # === 3) State + mask tails ===
    # Applied the same [A-Z]?X{5,} logic here to generalize from [SP]?X{6,}
    (r"\b[A-Z]{2}\s+[A-Z]?X{5,}\s+CARD\s+X{4}\b", " "), 
    (r"\b[A-Z]{2}\s+[A-Z]?X{5,}\b", " "), 

    # === 4) Dates/times === 
    (r"\b#?\d{2}[/-]\d{2}(?:[/-]\d{2,4})?\b", " "), # Dates like 10/23, 10/23/2025 
    (r"\b\d{1,2}\s+\d{2}\s+\d{2}\s*(?:AM|PM)\b", " "), # 10 23 25 PM 
    (r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)\b", " "), # Times like 10:23 AM 

    # === 4.5) Noise numbers (from n-grams) ===
    (r"\b(?:00|10|15|16|20|365)\b", " "),

    # === 5) Merchant-terminal boilerplate === 
    (r"\bMERCHANT\s+PURCHASE\s+TERMINAL\b\s*-?", " "), 
    (r"\bPOINT\s+OF\s+SALE\s+(?:WITHDRAWAL|DEBIT)\b\s*-?", " "), 
    (r"\b(?:CRD|ACH)\s+TRAN(?:\s+PPD(?:\s+ID)?)?\b", " "), 
    (r"\bCO\s+ID\s+\w+\s+(?:WEB|PPD)\b.*", " "), # Remove CO ID... 
     
    # === 6) Misc tails === 
    (r"\b(?:INST|PAYPAL)\s+XFER\b", " "), 
    (r"\b(?:XFER|WEB)\s+ID\b.*", " "), 
    (r"\b(?:ELECTRONIC|EXTERNAL)\s+WITHDRAWAL\b", " "), 
    (r"\bWITHDRAWAL\s+DEBIT\s+CARD\b(?:\s+DEBIT)?", " "), 
    (r"\bO(?:F)?\s+SALE\s+DEBIT\s+L\d{3}\b.*", " "), 
    (r"\b(?:ITEM|OVERDRAFT)\s+FEE\s+FOR\s+ACTIVITY\b.*", " "), 
    (r"\b(?:GENESIS[-\s]*FS\s+CARD\s+PAYMENT)\b", " "), 
    (r"\bBILL\s+PAYMENT\b", " "), 
    (r"\b(?:US|WA)\s+CARD\s+PURCHASE\b", " "), 
    (r"-\s*MEMO=", " "), 
    (r"(?:USA|US)$", " "), # Remove USA or US at the end 
    (r"\s+FSP$", " "), 
    (r"\bL\d{3}\b", " "), # Handle L340

    # === 7) Phone numbers === 
    (r"\b(?:\d{3}-\d{3}-\d{4}|XXX-XXX-XXXX)\b", " "), # 800-555-1212 
    (r"\b\d{3}-\d{4}\b", " "), # 555-1212 
    (r"\b(?:\d{3}\s*){1,2}\d{3}\s*\d{3,4}\b", " "), # 800 555 1212 or 1 800 555 1212 
    (r"\b#?\s*\d{3}-\d{3}-\d{1,4}\s*(?:AM|PM)?\b", " "), 

    # === 8) URLs/domains === 
    (r"^\.COM\s+BILL\b.*", " "), 
    (r"\.COM\b", " "), # Remove .COM anywhere
    (r"\s+COM$", " "), # Remove .COM at end of string

    # === 9) State abbreviations === 
    (STATE_REGEX, " "), # Remove standalone state codes 
    
    # === 9.5) City abbreviations ===
    (CITY_REGEX, " "), # Remove standalone city names
     
    # === 10) Noise Words (from 1-grams) ===
    (NOISE_WORDS_REGEX, " "),

    # === 11) Final Tidy (Punctuation) === 
    (r"[|%_=;\\/]+", " "), # Remove misc separators 
    (r"[-]{2,}", " "), # Collapse multiple hyphens
    (r"^\s*-\s*|\s*-\s*$", " "), # Remove leading/trailing hyphens
]

REGEX_POST = [
    re.compile(r".*(GODADDY\.COM|GODADDY)\b.*"),
    re.compile(r"^(AFTERPAY)\b.*"),
    re.compile(r"^(ALDI)\b.*"),
    re.compile(r"^(AMZN(?:\s*MKTP)?)\b.*"), # Handles amzn, amzn mktp
    re.compile(r"^(AMAZON(?:\.COM|\s+PRIME)?)\b.*"), # Handles amazon, amazon prime
    re.compile(r"^(APPLE(?:\.COM)?)\b.*"),
    re.compile(r"^(BRIGIT)\b.*"),
    re.compile(r"^(BURGER\s+KING)\b.*"),
    re.compile(r"^(CASH\s+APP)\b.*"),
    re.compile(r"^(CHICK-FIL-A)\b.*"),
    re.compile(r"^(CHIPOTLE)\b.*"),
    re.compile(r"^(CIRCLE\s+K)\b.*"),
    re.compile(r"^(COSTCO)\b.*"),
    re.compile(r"^(DAIRY\s+QUEEN)\b.*"),
    re.compile(r"^(DOLLAR\s+GENERAL)\b.*"),
    re.compile(r"^(DOLLAR\s+TREE)\b.*"),
    re.compile(r"^(DOORDASH)\b.*"),
    re.compile(r"^(DUNKIN)\b.*"),
    re.compile(r"^(EBAY)\b.*"), # Added from n-grams
    re.compile(r"^(ETSY)\b.*"),
    re.compile(r"^(FAMILY\s+DOLLAR)\b.*"),
    re.compile(r"^(FOOD\s+LION)\b.*"),
    re.compile(r"^(FRYS)\b.*"),
    re.compile(r"^(GOOGLE)\b.*"),
    re.compile(r"^(HELPPAY)\b.*"),
    re.compile(r"^(HOME\s+DEPOT)\b.*"),
    re.compile(r"^(INSTACART)\b.*"),
    re.compile(r"^(KFC)\b.*"),
    re.compile(r"^(KROGER)\b.*"),
    re.compile(r"^(LITTLE\s+CAESARS)\b.*"),
    re.compile(r"^(LOWE'?S)\b.*"),
    re.compile(r"^(LYFT)\b.*"),
    re.compile(r"^(MCDONALD'?S)\b.*"),
    re.compile(r"^(MICROSOFT)\b.*"),
    re.compile(r"^(PUBLIX)\b.*"),
    re.compile(r"^(ROSS)\b.*"),
    re.compile(r"^(SAFEWAY)\b.*"),
    re.compile(r"^(SAMS\s*CLUB|SAMSCLUB)\b.*"), # Updated from n-grams
    re.compile(r"^(SHOPRITE)\b.*"),
    re.compile(r"^(SONIC)\b.*"),
    re.compile(r"^(STARBUCKS)\b.*"),
    re.compile(r"^(SUBWAY)\b.*"),
    re.compile(r"^(TACO\s+BELL)\b.*"),
    re.compile(r"^(TARGET)\b.*"),
    re.compile(r"^(UBER(?:\s+EATS)?)\b.*"),
    re.compile(r"^(USPS)\b.*"),
    re.compile(r"^(VONS)\b.*"),
    re.compile(r"^(WALMART(?:\s*SUPERCENTER|\s*SUPER\s*C)?|WM\s*SUPERCENTER|WAL-MART|WAL\s*MART)\b.*"),
    re.compile(r"^(WENDY'?S)\b.*"),
    re.compile(r"^(7(?:-ELEVEN|\s+11))\s*\*?#?.*"),
    re.compile(r"^(DDBR)\s*\*?#?.*"),
]
