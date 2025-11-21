import pandas as pd, numpy as np,  matplotlib.pyplot as plt , seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, roc_curve, accuracy_score, auc
import lime.lime_tabular, lime, shap
from scipy.stats import spearmanr, pearsonr
import plotly.express as px
import warnings; warnings.filterwarnings('ignore')
import traceback as tb
class primary:
    def __init__(self):
                self.random =None
                self.models= {}
                self.X_train = None
                self.X_val = None
                self.y_train = None
                self.y_val = None
                self.explain_results={}
                self.fvs_scores={}   #Fairwashing scores
                self.shap_explanations ={}
                self.lime_explanations ={}
                self.train_raw = r'C:\Users\noahp\OneDrive - UTS\Honours Folder\UNSW_NB15_training-set.parquet'
                self.test_raw =r'C:\Users\noahp\OneDrive - UTS\Honours Folder\UNSW_NB15_testing-set.parquet'
    def main(self,random_val):
                 self.random_val = random_val
                 self.models = {}
                 self.explain_results = {}
                 self.fvs_scores = {}
                 self.shap_explanations = {}
                 self.lime_explanations = {}
                 model_configs={
                    "LogRegression":LogisticRegression(max_iter=1000,random_state=random_val),
                    "DecisionTree":DecisionTreeClassifier(random_state=random_val,max_depth=20, min_samples_leaf=4, min_samples_split=2), 
                    "XGB": XGBClassifier(booster='gbtree',n_estimators=100,max_depth=5, learning_rate=0.1, colsample_bytree=0.5, eval_metric='logloss'),
                    "MLP": MLPClassifier(hidden_layer_sizes=(64,32),max_iter=500, random_state=random_val,activation='relu',solver='adam')
                }
                 try:
                         self.train_data=pd.read_parquet(self.train_raw)
                         self.test_data=pd.read_parquet(self.test_raw)
                         print("Training set shape" + str(self.train_data.shape))
                         print("Test set shape" + str(self.test_data.shape))
                         print(self.train_data['label'].value_counts())
                         print(self.train_data['attack_cat'].value_counts())
                         X= self.train_data.drop(columns=['label','attack_cat'])
                         y_binary= self.train_data['label']
                         y_mc =self.train_data['attack_cat'] #multi class
                         
                         X_enc =pd.get_dummies(X, drop_first=False)
                         X_enc= X_enc.astype('float64') #compatible with Xgboost
                         
                         self.X_train, self.X_val, self.y_train,self.y_val=train_test_split(X_enc,y_binary, test_size=0.2, stratify=y_binary, random_state=random_val)
                         print("Features after Data Preparation  =  " + str(X_enc.shape[1]))
                         for model_name, model in model_configs.items():
                              try:
                                   print("Machine Learning training... " +model_name)
                                   model.fit(self.X_train,self.y_train)
                                   predicts=model.predict(self.X_val)
                                   accuracy = accuracy_score(self.y_val, predicts)
                                   recall = recall_score(self.y_val, predicts)
                                   precision = precision_score(self.y_val, predicts)
                                   fpr,tpr, thresholds = roc_curve(self.y_val,model.predict_proba(self.X_val)[:,1])
                                   roc_auc =auc(fpr,tpr)
                                   self.models[model_name]={ 'model':model,  'accuracy':accuracy, 'recall':recall,
                                        'precision':precision,   'roc_auc':roc_auc, 'predictions':predicts
                                   }
                                   print(f"{model_name} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, ROC AUC: {roc_auc:.4f}")
                              except Exception as e:
                                   print("ML Training Error "+str({model_name}))
                                   tb.print_exc()
                         return len(self.models) > 0
                 except FileNotFoundError as e:
                         print('Error file cannot be found')
                         return False     

                           
                
    def shap_section(self, model_name, size=100):
        model =self.models[model_name]['model']
        random_val=self.random_val
        X_sam =self.X_val.iloc[:size] #tesing sample size
        bg = self.X_train.sample(min(size,len(self.X_train)),random_state=self.random_val)    
       
        try:
             if len(X_sam.shape) ==1:
                  X_sam =X_sam.reshape(1,-1)
            
             if model_name =="LogRegression":
                explainer=shap.LinearExplainer(model, bg)
                shap_values=explainer.shap_values(X_sam)
                if isinstance(shap_values,list):
                        shap_values =shap_values[1] #Get positive class
            
             elif model_name in ["DecisionTree","XGB"]:          # Tree based methods explainer
                explainer =shap.TreeExplainer(model)
                shap_values= explainer.shap_values(X_sam)
                if isinstance(shap_values, list):
                    shap_values=shap_values[1]  
                elif len(shap_values.shape)==3:
                    shap_values = shap_values[:,:,1]   #if 3d shape

             elif model_name =="MLP":
                  explainer=shap.KernelExplainer(model.predict_proba,bg)
                  shap_values =explainer.shap_values(X_sam, nsamples=50)
                  if isinstance(shap_values, list):
                       shap_values=shap_values[1] # positive class
             if len(shap_values.shape) ==1:
                  shap_values= shap_values.reshape(1,-1)
             expected =self.X_train.shape[1]
             actual=shap_values.shape[1]
             
             if actual != expected:
                if actual>expected:
                    shap_values=shap_values[:, :expected]
                else:
                      padding = np.zeros((shap_values.shape[0], expected - shap_values.shape[1]))
                      shap_values = np.hstack([shap_values, padding])
             return shap_values,X_sam
        
        except Exception as e:
             print(f"Explainer error for model {model_name}")
             tb.print_exc()
             return None, None
        
    def feature_name_extraction(self,feature_name):
         #Check the Features names against columns
          for col_name in self.X_train.columns:
               if col_name in feature_name:
                    return col_name
          print(F"Error couldnt not find feature called {feature_name}")
          return None

    def lime_generation(self,model_name, indices=None,instances=100):
         'Lime generation'
         model = self.models[model_name]['model']
         if indices == None:
              indices =list(range(min(instances,len(self.X_val))))

         explainer =lime.lime_tabular.LimeTabularExplainer(training_data=self.X_train.values,
         feature_names=self.X_train.columns,class_names=['Benign','Attack'],mode='classification',
         discretize_continuous=True,sample_around_instance=True)
         lime_explanations=[]
         for idx in indices:
              try:
                    instance = self.X_val.iloc[idx].values 
                    lime_exp=explainer.explain_instance ( data_row=instance, predict_fn=model.predict_proba,
                    num_features=len(self.X_train.columns),num_samples=1000) # generate explanation
                    
                    feature_ranking=lime_exp.as_list() #Get feature importance
                    lime_importance= np.zeros(len(self.X_train.columns))
                    for feat_name,importance in feature_ranking:
                         base_feat_name = self.feature_name_extraction(feat_name)

                         if base_feat_name in self.X_train.columns:
                              feat_idx =self.X_train.columns.get_loc(base_feat_name)
                              lime_importance[feat_idx] += abs(importance)
                    lime_explanations.append(lime_importance)

              except Exception as e:
                   print(f'Error producing LIME explaination for {model_name} {idx}: {str(e)}')
                   lime_explanations.append(np.zeros(len(self.X_train.columns)))
         return np.array(lime_explanations)
    def normalise_data(self,shap_values, lime_values):
        shap_flat = shap_values.flatten()
        lime_flat = lime_values.flatten()
       
        shap_norm = np.linalg.norm(shap_flat)  #l2 normalization
        lime_norm = np.linalg.norm(lime_flat)
        shap_normalized =shap_flat / shap_norm if shap_norm > 0 else shap_flat
        lime_normalized = lime_flat/ lime_norm if lime_norm>0 else lime_flat
        return shap_normalized, lime_normalized
         
         #calculate narrative Flexibility score
    def get_binary_class(self,values):
          if len(values.shape) > 1 and values.shape[1] > 1: #check array dimensions 2+?
             if values.shape[1]==2:
                values =values[:,1] #if binary get positive classes
             else:
                  values =values[:,0]
          return values
    
    def narrative_flex_score(self, shap_values, lime_values, top_k=20):

        shap_values=self.get_binary_class(shap_values)
        lime_values=self.get_binary_class(lime_values)
        shap_norm, lime_norm =self.normalise_data(shap_values, lime_values)
        n_features = min (top_k,len(shap_norm)) # get the most important features

        shap_top_indices=  np.argsort(np.abs(shap_norm))[-top_k:]
        lime_top_indices=  np.argsort(np.abs(lime_norm))[-top_k:]
        shap_topset =set(shap_top_indices.tolist())
        lime_topset=set(lime_top_indices.tolist())

        #Find features that overlap
        overlap = len(shap_topset.intersection(lime_topset))
        overlap_ratio = overlap / n_features

        common_features =list(shap_topset.intersection(lime_topset))
        if len(common_features)>1:  # needs two or more for good comparison
             shap_ranks = [list(shap_top_indices).index(f) for f in common_features] 
             lime_ranks = [list(lime_top_indices).index(f) for f in common_features]
             try:
                  rank_corr, _ =spearmanr(shap_ranks,lime_ranks) # Calculate spearman correlation
                  rank_corr= abs(rank_corr)if  not np.isnan(rank_corr) else 0
             except:   
                    rank_corr =0
        else:
             rank_corr=0

        nfs = (1-overlap_ratio) *0.8+ (1-rank_corr) *0.2
        return min(max(nfs,0.0),1.0)

    def manipulation_potential_index(self,shap_values,lime_values):
        #Calculate MPI - measure the overal agrement and coorelatin of both methods"
        shap_values =self.get_binary_class(shap_values) #handle shap with multiple classes
        shap_norm,lime_norm =self.normalise_data(shap_values,lime_values)
        EPILSON= 1e-8
        if np.all(shap_norm == 0) and np.all(lime_norm ==0): #handle vectors with zero
             return 0.0
        
        #Calculate correlation of SHAP-LIME explanations
        non_zero_face = (np.abs(shap_norm)>EPILSON) | (np.abs(lime_norm)>EPILSON)
        if np.sum(non_zero_face) <2 :
             return 0.5 # Medium disagreement and maybe not enougt data

        try:
                filter_shap =shap_norm[non_zero_face]
                filter_lime= lime_norm[non_zero_face] 
                correlation,_ =pearsonr(filter_shap, filter_lime)
                correlation =abs(correlation) if not np.isnan(correlation) else 0
        except:
                correlation =0
        abs_diff = np.mean(np.abs(shap_norm-lime_norm)) #absolute difference of normalized SHAP-LIME
        mpi =(1-correlation) * 0.5 +abs_diff*0.5
        return min(max(mpi,0.0),1.0)
    
    def calculate_fairwashing_score(self,shap_values,lime_values,top_k=15):
        shap_values= self.get_binary_class(shap_values)
        shap_norm,lime_norm =self.normalise_data(shap_values,lime_values)
        nfs =self.narrative_flex_score(shap_values,lime_values)
        mpi=self.manipulation_potential_index(shap_values,lime_values)
        #Top k features
        shap_top_indices=  np.argsort(np.abs(shap_norm))[-top_k:]
        lime_top_indices=  np.argsort(np.abs(lime_norm))[-top_k:]
        shap_topset =set(shap_top_indices.tolist())
        lime_topset=set(lime_top_indices.tolist())
        feature_agreememt = len(shap_topset.intersection(lime_topset)) /top_k
        feature_deficit= 1- feature_agreememt
        fvs =(nfs * 0.35) + (mpi* 0.35 )+ (feature_deficit*0.3)
        return{ 
             'FVS': fvs, 'NFS': nfs, 'MPI': mpi, 
             'feature_agreement' :feature_agreememt, 'Feature Deficit' : feature_deficit
        }
    def assign_fairwashing_metrics(self, fvs_score):
      if fvs_score < 0.4: return "Low Risk"
      elif fvs_score < 0.65: return "Medium Risk"
      else: return "High Risk"
    
    def fairwashing_ml_application(self,instances=100) :
         print("Calculating Fairwashing benchmarks for ML models")
         for model_name in self.models.keys():
          if model_name not in self.shap_explanations or model_name not in self.lime_explanations:
              print(f"No explanations exist for {model_name}")
              continue
          print (f" Processing Model : {model_name}")
          shap_vals =self.shap_explanations[model_name]
          lime_vals=self.lime_explanations[model_name]
          instances_scores =[]

          for i in range(min(len(shap_vals), len(lime_vals))):
               shap_inst= shap_vals[i]
               lime_inst=lime_vals[i]
               scores=  self.calculate_fairwashing_score(shap_inst,lime_inst)       
               scores['instance']= i
               instances_scores.append(scores)
          
          results_df =pd.DataFrame(instances_scores)
          #Averages of results
          mean_fvs = results_df['FVS'].mean()
          std_fvs =results_df['FVS'].std()
          mean_nfs =results_df['NFS'].mean()
          mean_mpi =results_df['MPI'].mean()
          mean_feature_agree = results_df['feature_agreement'].mean()
          
          high_risk_sum = (results_df['FVS'] >=0.65).sum()
          medium_risk_sum =((results_df['FVS'] >=0.4) & (results_df['FVS'] <0.65 )).sum()
          low_risk_sum =(results_df['FVS']<0.4).sum()

          overall_risk =self.assign_fairwashing_metrics(mean_fvs)
          summary = { 'model': model_name, 'mean_fvs': mean_fvs, 'std_FVS': std_fvs, 'mean_nfs': mean_nfs,'mean_mpi': mean_mpi, 
                     'mean_feature_agree': mean_feature_agree, 'high_risk_instances': high_risk_sum, 'medium_risk_instances':medium_risk_sum,
                     'low_risk_instances': low_risk_sum, 'overall_risk': overall_risk}
            
          self.explain_results[model_name] = results_df
          self.fvs_scores[model_name]= summary
          print(f" Mean FVS {mean_fvs} and STD= {std_fvs:.3f} -> {overall_risk}")
          print(f" Narrative flexibility score {mean_nfs:.3f} ")
          print(f"Manipulation potential index {mean_mpi:.3f} ")
          print(f" Feature agreement {mean_feature_agree:.3f} ")
         return summary
     
    

     
    def visualization_agreement_heatmap(self,model_name,instances=50,top_k=20) :
     "Show agreememt between SHAP and LIME"
     if model_name not in self.shap_explanations:
          print(f'No explanations available for {model_name}')
          return
     shap_vals =self.shap_explanations[model_name][:instances]
     lime_vals=self.lime_explanations[model_name][:instances]

     no_of_features = len(self.X_train.columns) 

     agreement_matrix=[]
     for i in range(len(shap_vals)):
          shap_norm, lime_norm = self.normalise_data(shap_vals[i],lime_vals[i])
          shap_norm= shap_norm[:no_of_features] #Shrink to feature count
          lime_norm= lime_norm[:no_of_features]
          shap_top = np.argsort(np.abs(shap_norm))[-top_k:].tolist()
          lime_top = np.argsort(np.abs(lime_norm))[-top_k:].tolist()
     #Calculate agreement of each feature
          feature_agreement = []
          for j in range(no_of_features):
               if j in shap_top and j in lime_top:
                    feature_agreement.append(1)  #both agree
               elif j not in shap_top and j not in lime_top:
                    feature_agreement.append(0.5)  #both SHAP-LIME disagree
               else:
                    feature_agreement.append(0) #disagree
          agreement_matrix.append(feature_agreement)

     agreement_matrix =np.array(agreement_matrix)
     #Must check dimensions 
     if agreement_matrix.shape[1] != no_of_features:
          print(f" Error Agreement matrix has {agreement_matrix.shape[1]} features yet {no_of_features} was expected")
          agreement_matrix = agreement_matrix[:, :no_of_features]

     feature_variance =np.var(agreement_matrix, axis=0)
     no_of_features_to_plot = min(30, no_of_features)
     top_vary_features = np.argsort(feature_variance)[-no_of_features_to_plot:]

     top_vary_features = top_vary_features[top_vary_features< no_of_features]



     heatmap = px.imshow(agreement_matrix[:,top_vary_features].T,labels=dict(x='Instance Index', y='Feature', 
                    color='Agreement'), 
                    y=[self.X_train.columns[i] for i in top_vary_features], 
                    color_continuous_scale='RdYlGn',zmin=0,zmax=1,
                    color_continuous_midpoint=0.5,title=(f'{model_name}: SHAP & LIME Agreement over 100 Iterations '
                                                                                                       ))
     heatmap.show()






    def create_visualisations(self):
          if not self.fvs_scores:
               print("No values available for visualisations")
               return
          try:
               summary_data =[]
               for model_name, scores in self.fvs_scores.items():
                    summary_data.append({ 'Model':model_name,'FVS': scores['mean_fvs'],
                                        'NFS':scores['mean_nfs'], 'MPI':scores['mean_mpi'],'Feature_Agreement':scores['mean_feature_agree'],
                                        'Risk_Level':scores['overall_risk']

                    })
               summary_df=pd.DataFrame(summary_data)

               # Graph 2  bar chart
               
               plt.figure(figsize=(10,6))
               colors=['green' if x < 0.4 else 'orange' if x < 0.65 else 'red'
                         for x in summary_df['FVS']]
               bars = plt.bar(summary_df['Model'],summary_df['FVS'],color=colors,alpha=0.8 )
               plt.bar_label(bars,fmt='%.3f')
               plt.title('Fairwashing Vulnerability scores', fontweight='bold')
               plt.ylabel('FVS scores')  
                
               plt.axhline(y=0.4,color='orange',linestyle='--',alpha=0.7,label='Medium Risk')
               plt.axhline(y=0.65,color='red',linestyle='--',alpha=0.7,label='High Risk')
               plt.legend()
               plt.grid(True,alpha=0.3)
               plt.xticks(rotation=45)
               plt.tight_layout()
               plt.savefig('graph2_fvs_scores.png',dpi=300,bbox_inches='tight')
               plt.show()
               plt.close()


               #Graph 3 
               plt.figure(figsize=(10, 6))
               x = np.arange(len(summary_df))
               width = 0.25
               bars1=plt.bar(x - width, summary_df['NFS'],width,label='NFS', 
                         alpha=0.8,color='skyblue')
               
               bars2=plt.bar(x, summary_df['MPI'], width, label='MPI',
                         alpha=0.8,color='lightcoral')
               bars3=plt.bar(x + width, summary_df['Feature_Agreement'], 
                         width,label='Feature Agreement',alpha=0.8, color='lightgreen')
               
               plt.bar_label(bars1,fmt='%.3f')
               plt.bar_label(bars2,fmt='%.3f')
               plt.bar_label(bars3,fmt='%.3f')
               plt.title('Fairwashing Components by Model',fontweight='bold', fontsize=14)
               plt.xlabel('Model')
               plt.ylabel('Score')
               plt.xticks(x,summary_df['Model'],rotation=45)
               plt.legend()
               plt.tight_layout()
               plt.savefig('graph3_component_breakdown.png',dpi=300,bbox_inches='tight')
               plt.show()     
               plt.close()


               #graph 4 
               plt.figure(figsize=(10, 6))
               risk_data = []
               for model_name in self.fvs_scores.keys():
                    scores= self.fvs_scores[model_name]
                    risk_data.append([
                         scores['low_risk_instances'],scores['medium_risk_instances'],
                         scores['high_risk_instances']])
          
               risk_df = pd.DataFrame(risk_data, index=list(self.fvs_scores.keys()),
                                   columns=['Low Risk','Medium Risk','High Risk'])
          
               sns.heatmap(risk_df, annot=True,fmt='d',
                           cmap='RdYlGn_r',cbar_kws={'label': 'Number of Instances'})
               plt.title('Risk Distribution of models',fontweight='bold',fontsize=14)
               plt.ylabel('Model',fontsize=12)
               plt.tight_layout()
               plt.savefig('graph4 risk_heatmap.png',dpi=300,bbox_inches='tight')
               plt.show()

               #Graph 5 
               plt.figure(figsize=(10, 6))
               bars = plt.bar(summary_df['Model'],summary_df['Feature_Agreement'],color='mediumpurple',alpha=0.7)
               plt.bar_label(bars,fmt="%.3f")
               plt.title('Feature Agreement by model',fontweight='bold',fontsize=14)
               plt.ylabel('Agreement Score ',fontsize=12)
               plt.ylim(0,1)
               plt.grid(True,alpha=0.3)
               plt.xticks(rotation=45)
               plt.tight_layout()
               plt.savefig('graph5 feature_agreement.png',dpi=300,bbox_inches='tight')
               plt.show()
          except Exception as e:
               print("Error in generating Visualisations")
                 





    def run_pipeline(self, random_val=42, size=100):
         print("Starting Explanation Framework")
     
          # Phase 1 Prepare data and Train models   
         print("\n Preparing ML Models ")
         if not self.main(random_val):
              print("Model Training has failed!\n")
              return False
         print(f" Models Successfully trained \n")

     #Phase 2 Generate Shap explanations
         print("Phase 2 Generating Shap Explanations ")
         for model_name in self.models.keys():
              try: 
                   shap_vals, X_sample =self.shap_section(model_name, size=size)
                   if shap_vals is not None:
                        self.shap_explanations[model_name] = shap_vals
                        print(f" SHAP generated for {model_name}")
              except Exception as e:
                   print(f"Error SHAP generation failed for : {model_name}")
                   tb.print_exc()
          
     #Phase 3 Generate Lime explanations
         print("Phase 3 Generating LIME Explanations ")
         for model_name in self.models.keys():
              try: 
                   lime_vals =self.lime_generation(model_name, instances=size)
                   if lime_vals is not None:
                        self.lime_explanations[model_name] = lime_vals
                        print(f"LIME generated for {model_name}")
              except Exception as e:
                   print(f"Error LIME generation failed for : {model_name}")
                   tb.print_exc()
     #Phase 4 Fairwashing Framework Begins 
         print("\n Phase 4 Calculating Fairwashing Vulnerability Scores")
         self.fairwashing_ml_application(instances=size)

     #Phase 5 Show results 
         print("\n Phase 5 Generating Visuals")
         self.create_visualisations()
         for model_name in self.models.keys():
              if model_name in self.shap_explanations and model_name in self.lime_explanations:
                  try:
                       print(f"Visualisation for {model_name}")
                      # self.visualise_explanations(model_name, instance=0, top_k=15)
                       self.visualization_agreement_heatmap(model_name, instances=size,top_k=20)
                       
                  except:
                       print(f" Visualization of graphs failed for {model_name}") 
     #Phase 6 Results Summary
         print("Fairwashing Framework Summary")
         for model_name, scores in self.fvs_scores.items():
              print(f"\n{model_name}:")
              print(f" Mean FVs: {scores['mean_fvs']:.3f}  ({scores['overall_risk']})")
              print(f" NFS: {scores['mean_nfs']:.3f}")
              print(f" MPI: {scores['mean_mpi']:.3f}")
              print(f" Feature Agreement {scores['mean_feature_agree']:.3f}")
         return True
          


if __name__=="__main__":
    obj =primary()
    obj.run_pipeline(random_val=42,size=100)