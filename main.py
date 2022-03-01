from flask import Flask, request, render_template
from chatbot import chatbot_response
app = Flask('app')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    bot_response=chatbot_response(user_input)
    print("Lila: " + bot_response)
    return render_template('index.html', user_input=user_input,
            bot_response=bot_response)
    

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=9696)