import tkinter as tk
import mlmodels
from sklearn.externals import joblib
import dataoperations as dataop
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
filename = 'finalized_model.sav' #where the loaded model is kept

HEIGHT = 500
WIDTH = 600

def optionMenuSelection(x): #the function that is called when there is a selection from the option menu
    global entry_ready #the variable that has the processed entry and it is ready to be predicted
    entry_ready = mlmodels.classifiers(x,entry.get())
    
def buttonPressed(): #the function that is called when the button is pressed, it prints out the output
    loaded_model = joblib.load(filename)  #Previously saved model is loaded here  
    prediction = loaded_model.predict(dataop.cv_text_test)
    print(prediction)
    
    score = accuracy_score(prediction, dataop.y_test)
    report = classification_report(dataop.y_test, prediction, target_names=['0','1'])
    print(report)
    
    entry_prediction = loaded_model.predict(entry_ready)
    print(entry_prediction)
    
    cm = confusion_matrix(dataop.y_test, prediction, labels=[0,1])
    print(cm)
    #false positive çok fazla olduğundan precision düşük olmalı??
    
    if int(entry_prediction) == 1:
        labelText = "This is a positive entry with " + str(round(score,2)) + " accuracy"
    else:
        labelText = "This is a negative entry with " + str(round(score,2)) + " accuracy"

    label.configure(text= labelText)
    
    #score = mlmodels.cross_val_score(loaded_model, dataop.cv_text , dataop.y, cv = mlmodels.split, scoring='accuracy', n_jobs=-1)

#GUI widgets, their initiations and adjustments
root = tk.Tk()

canvas = tk.Canvas(height=HEIGHT, width=WIDTH)
canvas.pack()

frame = tk.Frame(root, bg = '#D6EAF8')
frame.place(relx=0.1, rely = 0.1,relwidth =0.8, relheight=0.8)

entry = tk.Entry(frame, font=20)
entry.place(relx = 0.1, rely=0.05, relwidth = 0.8, relheight=0.4)

clicked = tk.StringVar(root)
clicked.set("Select the algorithm")

drop = tk.OptionMenu(frame, clicked, "Logistic Regression", "Linear SVM", "Multinominal NB","Random Forest", "Neural Networks", command = lambda x: optionMenuSelection(entry.get()))
drop.place(relx = 0.35, rely=0.5)

button = tk.Button(frame, text = "Submit", command = buttonPressed)
button.place(relx =0.43, rely=0.65)

label = tk.Label(frame, bg = '#D6EAF8')
label.place(relx = 0.3, rely = 0.8)

root.mainloop()