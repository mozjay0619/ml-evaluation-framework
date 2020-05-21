import datetime

def check_date_format(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d').date()
        return True
    except ValueError:
        return False