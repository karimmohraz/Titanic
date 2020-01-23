import json
import numpy as np
import pandas as pd
import pickle

# Global vars to keep track of model status
serlzd_model = None
model_ready = False

def preprocess_text_input(data):
    # text input
    # 	"sex": sSex,
    # 	"embarked": "S",
    # 	"class": sClass,
    # 	"who": sWho,
    # 	"adult_male": sAdultMale,
    # 	"deck": null,
    # 	"embark_town": "Southhampton",
    # 	"alone": sAlone,
    # 	"pclass": iPClass,
    # 	"age": iAge,
    # 	"sibsp": iSibsp,
    # 	"parch": iParch,
    # 	"fare": 1
    #
    # convert to integer columns: pclass sex age sibsp fare embarked who adult_male alone
    
    # remove unused columns 
    df = pd.Dataframe(data, columns = ['sex', 'embarked', 'class', 'who, 'adult_male, 'deck', 'embark_town', 'alone', 'pclass', 'age', 'sibsp', 'parch', 'fare'])
    df = df.drop(['class', 'deck', 'embark_town', 'parch'], axis=1)
    
    # convert non-numeric data
    genders = {"male": 0, "female": 1}
    df['sex'] = df['sex'].map(genders)
    
    ports = {"S": 0, "C": 1, "Q": 2}
    df['embarked'] = df['embarked'].map(ports)

    alive = {"yes": 1, "no": 0}
    df['alive'] = df['alive'].map(alive)
    
    who = {"man": 1, "woman": 2, "child": 0}
    df['who'] = df['who'].map(who)
    
    boool = {True: 1, False: 0}
    df['alone'] = df['alone'].map(boool)
    df['adult_male'] = df['adult_male'].map(boool)

    df['age'] = df['age'].astype(int)
    df.loc[ df['age'] <= 11, 'age'] = 0
    df.loc[(df['age'] > 11) & (df['age'] <= 18), 'age'] = 1
    df.loc[(df['age'] > 18) & (df['age'] <= 22), 'age'] = 2
    df.loc[(df['age'] > 22) & (df['age'] <= 27), 'age'] = 3
    df.loc[(df['age'] > 27) & (df['age'] <= 33), 'age'] = 4
    df.loc[(df['age'] > 33) & (df['age'] <= 40), 'age'] = 5
    df.loc[(df['age'] > 40) & (df['age'] <= 66), 'age'] = 6
    df.loc[ df['age'] > 66, 'age'] = 6
    
    df['fare'] = df['fare'].astype(int)
    df.loc[ df['fare'] <= 7.91, 'fare'] = 0
    df.loc[(df['fare'] > 7.91) & (df['fare'] <= 14.454), 'fare'] = 1
    df.loc[(df['fare'] > 14.454) & (df['fare'] <= 31), 'fare']   = 2
    df.loc[(df['fare'] > 31) & (df['fare'] <= 99), 'fare']   = 3
    df.loc[(df['fare'] > 99) & (df['fare'] <= 250), 'fare']   = 4
    df.loc[ df['fare'] > 250, 'fare'] = 5
    
    # reorder columns to match NN input
    df = df[['pclass', 'sex', 'age', 'sibsp', 'fare', 'embarked', 'who', 'adult_male', 'alone']]
    
return df.to_numpy()

    
# Validate input data is JSON
def is_json(data):
  try:
    json_object = json.loads(data)
  except ValueError as e:
    return False
  return True

# When Model Blob reaches the input port
def on_model(model_blob):
    global serlzd_model
    global model_ready

    serlzd_model = model_blob
    model_ready = True
    api.logger.info("Model Received & Ready")
    
# Client POST request received
def on_input(msg):
    error_message = ""
    success = False
    try:
        api.logger.info("POST request received from Client - checking if model is ready")
        if model_ready:
            api.logger.info("Model Ready")
            api.logger.info("Received data from client - validating json input")
            
            user_data = msg.body.decode('utf-8')
            # Received message from client, verify json data is valid
            if is_json(user_data):
                api.logger.info("Received valid json data from client - ready to use")
                
                feed = json.loads(user_data)
                input_data = np.array(feed['data'])
                api.logger.info(str(input_data))
                input_data = preprocess_text_input(input_data)
                
                model_dict = pickle.loads(serlzd_model)
                mlp = model_dict['model']
                scaler = model_dict['scaler']
                input_data = scaler.transform(input_data)
                
                # check path
                attr = msg.attributes
                op_id = attr['openapi.operation_id']
                api.logger.info('operation_id: ' + op_id )
                if 'predict_classes' in op_id:
                    prediction = mlp.predict(input_data).tolist()
                else:
                    prediction = mlp.predict_proba(input_data).tolist()
                api.logger.info(str(prediction))

                success = True
            else:
                api.logger.info("Invalid JSON received from client - cannot apply model.")
                error_message = "Invalid JSON provided in request: " + user_data
                success = False
        else:
            api.logger.info("Model has not yet reached the input port - try again.")
            error_message = "Model has not yet reached the input port - try again."
            success = False
    except Exception as e:
        api.logger.error(e)
        error_message = "An error occurred: " + str(e)
    
    if success:
        # apply carried out successfully, send a response to the user
        msg.body = json.dumps({'Results': prediction})
    else:
        msg.body = json.dumps({'Error': error_message})
    
    new_attributes = {'message.request.id': msg.attributes['message.request.id']}
    msg.attributes =  new_attributes
    api.send('output', msg)
    
api.set_port_callback("modelblob", on_model)
api.set_port_callback("input", on_input)
