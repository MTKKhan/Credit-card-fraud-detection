import pickle
from flask import Flask,render_template,request
import numpy as np
app = Flask(__name__)

# model = pickle.load(open('model.pkl','rb'))
with open('lor_model.pkl', 'rb') as f:
    lor_model = pickle.load(f)
with open('rfc_model.pkl', 'rb') as f:
    rfc_model = pickle.load(f)
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

print(lor_model[1])
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    time = float(request.form['time'])
    temp = request.form['val']
    val = temp.split()
    values = []
    for i in range(len(val)):
        values.append(float(val[i]))
    values.insert(0,time)
    final = np.array([values])

    # model and variations.
    model = str(request.form['model'])
    variation = int(request.form['variation'])
    if model == 'lor':
        print(model)
        predicting = lor_model[variation].predict(final)
    elif model == 'rfc':
        predicting = rfc_model[variation].predict(final)
    elif model == 'lr':
        predicting = lr_model[variation].predict(final)

    if predicting == 1:
        return render_template('temp.html',predict="FRAUD TRANSACTION")
    else:
        return render_template('temp.html',predict="LEGIT TRANSACTION")
    
if __name__ == '__main__':
    app.run(debug=True,port=8000)


