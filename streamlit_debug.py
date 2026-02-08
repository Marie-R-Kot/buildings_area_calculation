"""
Этот файл позволяет запустить streamlit_app.py в отладочном режиме через VS Code / PyCharm

Документация:
https://stackoverflow.com/questions/60172282/how-to-run-debug-a-streamlit-application-from-an-ide
"""
from streamlit.web.bootstrap import run

app_name = 'streamlit_app.py'
run(app_name, False, [], {})
