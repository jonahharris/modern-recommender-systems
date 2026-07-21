# Figure - Listing 7.10: Generating a time-of-day context token
# Source: chapters/ch09.md lines 349-360
# Chapter: 9
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
from datetime import datetime

def get_time_token(timestamp):
  hour = datetime.fromtimestamp(timestamp).hour
  if 6 <= hour < 12:
    return "TIME_MORNING"
  elif 12 <= hour < 18:
    return "TIME_AFTERNOON"
  elif 18 <= hour < 23:
    return "TIME_EVENING"
  else:
    return "TIME_NIGHT"
