import pandas as pd
import holidays
from datetime import timedelta

def add_custom_holidays(years):
    custom = {}
    for y in years:
        # Find Thanksgiving (4th Thursday of November)
        thanksgiving = [
            d for d, name in holidays.UnitedStates(years=[y]).items()
            if "Thanksgiving" in name
        ][0]

        # Custom additions
        black_friday = thanksgiving + timedelta(days=1)
        cyber_monday = black_friday + timedelta(days=3)

        custom[black_friday] = "Black Friday"
        custom[cyber_monday] = "Cyber Monday"
    return custom



# helper function to calculate distance from holidays
def holiday_context(date, holiday_dates, holiday_names):
    """Compute holiday context using globally pre-sorted dates."""
    
    # find prev and next index using search
    prev_idx = max([i for i, h in enumerate(holiday_dates) if h <= date], default=None)
    next_idx = min([i for i, h in enumerate(holiday_dates) if h >= date], default=None)

    days_since_prev = (date - holiday_dates[prev_idx]).days if prev_idx is not None else None
    days_until_next = (holiday_dates[next_idx] - date).days if next_idx is not None else None
    
    prev_name = holiday_names[prev_idx] if prev_idx is not None else None
    next_name = holiday_names[next_idx] if next_idx is not None else None

    return pd.Series({
        'days_since_prev_holiday': days_since_prev,
        'days_until_next_holiday': days_until_next,
        'prev_holiday': prev_name,
        'next_holiday': next_name
    })