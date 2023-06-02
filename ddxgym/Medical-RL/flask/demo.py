import gym, uuid, json, os
from flask import Flask, render_template, request, session
from flask_session import Session
from flask_apscheduler import APScheduler
import datetime

app = Flask(__name__)
app.secret_key = "31649a82ad564b9ba804eab0e02a82bfd5525079c6eeff0c56f10d4eda3f7022"
app.config['SECRET_KEY'] = "31649a82ad564b9ba804eab0e02a82bfd5525079c6eeff0c56f10d4eda3f7022"
app.config['SESSION_TYPE'] = 'filesystem'
scheduler = APScheduler()

scheduler.api_enabled = True
scheduler.init_app(app)
scheduler.start()
Session(app)
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/demo", methods=['GET', 'POST'])
def demo():
    
    num_diseases = int(request.form.get("num_diseases", 256))
    session["num_diseases"] = num_diseases  
    session["cheating_level"] = 0
    if not "gym_id" in session:
        session["gym_id"] = str(uuid.uuid4())
        run_date = datetime.datetime.now() + datetime.timedelta(days=1)
    
    env = gym.make("gym_medical:doctorsim-v0", 
                    data_path=os.getenv("MEDRL_DATA_PATH"), 
                    max_diseases = num_diseases,
                    tokenizer="bert-base-uncased",
                    is_csv= True if "csv" in os.getenv("MEDRL_DATA_PATH") else False,
                    observation_length=512,
                    )
    
    env.action_map = {env.procedures[a].name.title(): i for i, a in enumerate(env.procedures_to_actions)}
    session["actions"] = json.dumps(list(env.action_map.keys()))
    
    obs = env.reset()
    set_session_env(env, disease_info=env.render_html(), procedure_info=get_procedure_table(env))
    return render_template("demo.html")
 

@app.route("/demo/new", methods=['GET', 'POST'])
def new_patient():
    
    env = session["env"]
    obs = env.reset()
    
    set_session_env(env, disease_info=env.render_html(), procedure_info=get_procedure_table(env))
    
    return render_template("demo.html")

@app.route("/demo/step", methods=['POST'])
def step():
    global app
    
    env = session["env"]
    
    obs, r, done, info = env.step(env.action_map[request.form.get("action")])
    
    if done:
        print(f"app root path: {app.root_path}")
        with open(os.path.join(app.root_path, 'static/episode_log.csv'), "a") as f:
            p = env.get_patient()
            f.write(f"{datetime.datetime.now()},{r},{r>500},{p.disease.name}, {p.steps_alive}, {[prod.name for prod in p.applied_procedures]}\n")
    
    set_session_env(env, r, done, info, env.render_html())
    
    print(session)
    return render_template("demo.html")

def set_session_env(env, r = None, done = None, info = None, disease_info = None, procedure_info = None):
    session["env"] = env
    session["last_obs"] = env.get_patient().get_representation().replace(".", ".<br>")
    session["last_r"] = r
    session["last_done"] = done
    session["last_info"] = info
    session["disease_info"] = disease_info
    
    if procedure_info is not None:
        session["procedure_info"] = procedure_info
    
            
def get_procedure_table(env):
    d = env.get_patient().disease
    table = []
    for s in d.symptoms.values():
        s = s["symptom"]
        symptom = s.name
        examinations = ", ".join(f"{e.name}({env.procedures_to_actions.index(e.id)})" for e in s.examinations)
        treatments = ", ".join(f"{t.name}({env.procedures_to_actions.index(t.id)})" for t in s.treatments)
        is_main = s.is_main
        table += [{"symptom": symptom,
                   "examinations": examinations,
                   "treatments": treatments,
                   "is_main": is_main
                }]    
    return table   
if __name__ == '__main__':
    app.run()