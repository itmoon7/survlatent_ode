 
See [Notebook example](https://github.com/itmoon7/survlatent_ode/blob/main/notebook_example.ipynb) for a time to a single event prediction using [Framingham dataset](https://biolincc.nhlbi.nih.gov/media/teachingstudies/FHS_Teaching_Longitudinal_Data_Documentation_2021a.pdf?link_time=2022-02-03_18:20:47.023970). We'll release more iPython examples. Stay tuned!

### SurvLatent ODE

Abstract : Effective learning from electronic health records (EHR) data for prediction of clinical outcomes is often challenging because of features recorded at irregular timesteps and loss to follow-up as well as competing events such as death or disease progression. To that end, we propose a generative time-to-event model, SurvLatent ODE, which adopts an Ordinary Differential Equation-based Recurrent Neural Networks (ODE-RNN) as an encoder to effectively parameterize a latent representation under irregularly sampled data. Our model then utilizes the latent representation to flexibly estimate survival times for multiple competing events without specifying shapes of event-specific hazard function. We demonstrate competitive performance of our model on MIMIC-III, a freely-available longitudinal dataset collected from critical care units, on predicting hospital mortality as well as the in-house data on predicting onset of Deep Vein Thrombosis (DVT), a life-threatening complication for patients with cancer, with death as a competing event. SurvLatent ODE outperforms the current clinical standard Khorana Risk scores for stratifying DVT risk groups.

![alt text](https://github.com/itmoon7/survlatent_ode/blob/main/survlatent_ode_architecture.png?raw=true)

#### Model architecture of SurvLatent ODE: 
The ODE-RNN encoder parametrizes a patient-specific temporal trajectory of covariates $x_i(0),...,x_i(p)$ into the latent embedding and describes approximate posterior over the initial latent variable $z_{i,0}$. The sampled initial latent variable $z_{i,0}$ is then decoded into the latent trajectory, $z_i^{t_m}  = (z_{i,1}, z_{i,2}, ..., z_{i,t_m})$ by calling a black-box differential equation solver ODESolve$(g_{\phi}, z_{i,0}, t = 0, ...,t_m)$. Finally, $m_\beta(\cdot)$, which consists of cause-specific decoder modules, subsequent fully connected layer, and softmax layer, maps the corresponding latent trajectory to estimate cause-specific hazard function for each event.

#### Utilized Python packages
torch 1.11.0 <br>
torchdiffeq 0.2.3 <br>
tqdm 4.63.0 <br>
scikit-learn 1.0.2 <br>
scikit-survival 0.17.2 <br>
pandas 1.3.5 <br>
numpy 1.19.5 <br>
lifelines 0.38.0

<!-- <img src="https://render.githubusercontent.com/render/math?math=x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2b}"> -->
#### Citation :
@misc{moon2022survlatent,
      title={SurvLatent ODE : A Neural ODE based time-to-event model with competing risks for longitudinal data improves cancer-associated Deep Vein Thrombosis (DVT) prediction}, 
      author={Intae Moon and Stefan Groha and Alexander Gusev},
      year={2022},
      eprint={2204.09633},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

