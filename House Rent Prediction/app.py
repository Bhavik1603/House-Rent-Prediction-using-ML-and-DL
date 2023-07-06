import os
import pickle
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn import linear_model
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")
target_classes = ins.select_target_classes(couch=True,dining_table=True,bench=True,refrigerator=True,bed=True,oven=True,microwave=True,toaster=True,clock=True,tv=True)

data = pd.read_csv('GoregaonData.csv')
#loading lasso model
pipe = pickle.load(open('LassoModel.pkl', 'rb'))


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    societies = sorted(data['Society'].unique()) 
    locations = sorted(data['Loc'].unique())
    return render_template('index.html', societies=societies, locations=locations)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        gym = request.form['gym']
        lift= request.form['lift']
        swimming_pool = request.form['swimming_pool']
        kitchen = request.files['kitchen']
        hall = request.files['hall']
        bedroom = request.files['bedroom']
        
        # if user does not select file, browser also submits an empty part without filename
        if kitchen.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if kitchen and allowed_file(kitchen.filename):
            filenameOfKitchen = secure_filename(kitchen.filename)
            pathOfKitchenFile = os.path.join(app.config['UPLOAD_FOLDER'], filenameOfKitchen)
            print(pathOfKitchenFile)
            kitchen.save(pathOfKitchenFile)

        if hall.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if hall and allowed_file(hall.filename):
            filenameOfHall = secure_filename(hall.filename)
            pathOfHallFile = os.path.join(app.config['UPLOAD_FOLDER'], filenameOfHall)
            hall.save(pathOfHallFile)
        
        if bedroom.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if bedroom and allowed_file(bedroom.filename):
            filenameOfBedroom = secure_filename(bedroom.filename)
            pathOfBedroomFile = os.path.join(app.config['UPLOAD_FOLDER'], filenameOfBedroom)
            bedroom.save(pathOfBedroomFile)
        print("All pics Saved")
        resultsOfKitchenImage, output1 = ins.segmentImage("{}".format(pathOfKitchenFile), segment_target_classes= target_classes,show_bboxes=True, output_image_name="{}".format(pathOfKitchenFile))
        li1 = resultsOfKitchenImage['class_names']
        print(type(li1))
        countk=len(li1)
        resultsOfHallImage, output2 = ins.segmentImage("{}".format(pathOfHallFile),segment_target_classes= target_classes, show_bboxes=True, output_image_name="{}".format(pathOfHallFile))
        li2 = resultsOfHallImage['class_names'] 
        counth=len(li2)
        resultsOfBedroomImage, output3 = ins.segmentImage("{}".format(pathOfBedroomFile),segment_target_classes= target_classes, show_bboxes=True, output_image_name="{}".format(pathOfBedroomFile))
        li3 = resultsOfBedroomImage['class_names']
        countb=len(li3)
        total_furniture=counth+countk+countb
        furni=0
        if total_furniture > 15:
            furni = 1
        elif total_furniture <= 15 and total_furniture >= 2:
            furni = 0.5
        elif total_furniture == 0:
            furni = 0
        if request.method == 'POST':
            location = request.form['Location']
            bhk = request.form['bhk']
            sqft = request.form['sqft']
            bath = request.form['bath']
            powe = 1
            input = pd.DataFrame([[location, sqft, bhk,bath,furni,lift,swimming_pool,gym,powe]],columns = ['Loc','Carpet Area','BHK','Bath','Furnished','Lift','Swimming Pool','Gym','Power Back Up']) 
            prediction = pipe.predict(input)[0]
            print(prediction)
        return render_template('predict.html', li1 = li1, li2 = li2, li3 = li3, swimming_pool=swimming_pool,lift=lift,gym=gym,countk=countk,counth=counth,countb=countb,total_furniture=total_furniture,prediction=prediction)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
