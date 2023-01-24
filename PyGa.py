#---------------------------------------------------#
#--- Creating a K-fold cross validation function ---#
#---------------------------------------------------#

def CV_grid (X, y, mod, nfolds, seed, verbose, over, under, i_folds):
  rd.seed(seed)
  from sklearn import metrics
  from collections import Counter
  from imblearn.over_sampling import RandomOverSampler
  from imblearn.under_sampling import RandomUnderSampler
  from imblearn.over_sampling import SMOTE  
  X1, y1 = np.copy(X), np.copy(y) 
  #X1, y1 = X.copy(deep=True), y.copy(deep = True)
  X1 = pd.DataFrame(X1)
  y1 = pd.DataFrame(y1)  
  p = X1.shape[1]
  def seq (n):
    seq1 = []
    x = 0
    for i in range(0,n):
      seq1.append(x + i) 
    return seq1
  X1.index = seq(len(y))
  y1.index = seq(len(y))
  ind = rd.choices(seq(nfolds), k = X.shape[0])
  prec = []
  spec = []
  sen = []
  auc = []
  auc_pr_rec = []
  f1 = []
  X1['col1'] = ind
  y1['col1'] = ind
  
  for i in range(0,i_folds):
    Xt, yt = X1.loc[X1['col1']!=i,], y1.loc[y1['col1']!=i,]
    Xv, yv = X1.loc[X1['col1']==i,] , y1.loc[y1['col1']==i,]
    Xt, Xv = Xt.iloc[:,:p],Xv.iloc[:,:p]
    yt, yv = yt.iloc[:,0], yv.iloc[:,0]
    
    if under > 0:
      rus = RandomUnderSampler(sampling_strategy=under)
      Xt, yt = rus.fit_resample(Xt, yt)
      #Xv, yv = rus.fit_resample(Xv, yv)
          
    if over > 0:
      smo = SMOTE(sampling_strategy=over)
      Xt, yt = smo.fit_resample(Xt, yt)
        
    mod.fit(Xt, yt)
    y_pred=mod.predict(Xv)
    y_prob = mod.predict_proba(Xv)
    y_prob = y_prob[:,1]
    
    tn, fp, fn, tp = metrics.confusion_matrix(yv, y_pred).ravel()
    prec.append (metrics.precision_score(yv, y_pred, zero_division = 0))
    spec.append (tn / (tn+fp))
    sen.append (tp / (tp+fn))
    fpr, tpr, thresholds = metrics.roc_curve(yv, y_prob, pos_label=1)
    auc.append(metrics.auc(fpr, tpr))
    precision, recall, thresholds = metrics.precision_recall_curve(yv, y_prob)
    auc_pr_rec.append (metrics.auc(recall, precision))
    f1.append(metrics.f1_score(yv, y_pred, pos_label=1, average='binary'))

    if verbose == True:
      print("Class number on the training data:", Counter(yt))
      print("Class number on the testing data:",  Counter(yv))
      #print("number of features: ", Xt.shape[1])
      print('Fold:',i+1, "done...")
      print("Area under Prec-Recall curve:", auc_pr_rec[i])
      #print("AUC:", auc[i])
  
  del X1
  del y1
  result = [prec, spec, sen, auc, auc_pr_rec, f1]
  return result
  
#############################################################

#---------------------------------------------------#
#--- A simple GA for hyperparameter fine-tunning ---#
#---------------------------------------------------#
def gen_alg (ngen, popsize, mut_rate, elitism, cross_rate, tsize, fit_fun, verbose, bin_size):  
  from scipy.stats import binom
  import math
  import random 
  pop_average = []
  best_average = []
  population = []
  gen = 0

  #Nested functions
  def sample(start, size, n):
    import random as rd
    if (n >= size):
      return "Error n must be smaller than size"
    random_list = np.empty(n)
    random_list[0] = int(rd.randint(start,size))
    for i in range(n-1):
      n2 = int(rd.randint(start,size))
      while n2 in random_list:
        n2 = int(rd.randint(start,size))
      random_list[i+1] = n2
    return random_list.astype(int)

  #The crossing over function 
  def cross_over (population, mut_rate, cross_rate, tsize, elitism, score):
    popsize = len(population)    
    children = []    
    for i in range(math.ceil(popsize/2)):
      i1 = sample(0, popsize-1, tsize)
      i2 = sample(0, popsize-1, tsize)
      while any(x in i1 for x in i2):
        i2 = sample(0, popsize-1, tsize)
      candidates1 = []
      candidates2 = []
      score1 = []
      score2 = []      
      for i in range(tsize):
        index1 = i1[i]
        index2 = i2[i]
        candidates1.append(population[index1]) 
        candidates2.append(population[index2]) 
        score1.append(score[index1])
        score2.append(score[index2])
      best1 = max(range(len(score1)), key=score1.__getitem__)
      best2 = max(range(len(score2)), key=score2.__getitem__)
      p1 = candidates1[best1]
      p2 = candidates2[best1]
      #two-point crossing over
      points = sample(1,bin_size-1,2)
      points.sort()
      do_cross = binom.rvs(1, cross_rate, size = 1)
      if(do_cross==1):
        child1 = np.append(np.append(p1[0:(points[0])],p2[points[0]:points[1]]), p1[points[1]:len(p1)])
        child2 = np.append(np.append(p2[0:(points[0])],p1[points[0]:points[1]]), p2[points[1]:len(p1)])
      else:
        child1 = p1
        child2 = p2
      #Bit flip mutation
      mut1 = binom.rvs(1, mut_rate, size = bin_size)
      mut2 = binom.rvs(1, mut_rate, size = bin_size)      
      child1 = abs(child1 - mut1)
      child2 = abs(child2 - mut2)
      children.append(child1)
      children.append(child2)
    children = children[0:(popsize)]
    return (children)
  #Create the population for generation 0  
  for i in range (popsize):
    population.append(np.array(binom.rvs(1, 0.5, size = bin_size)))
  score = []
  #Compute Scores for Generation 0
  for z in range(popsize):
    score.append (fit_fun(population[z]))    
  score = np.array(score)  
  best =  score.argsort()[-elitism:][::-1] #Get the index of best n individuals according to elitism 
  pop_average.append (np.mean(score))
  best_average.append (np.mean(score[best]))
  if (verbose == True):
      print('Generation:', gen)
      print('Population average:', pop_average[gen])
      print('Best:', best_average[gen])    
  #gen+=1
  #Loop for the number of generations
  while gen < ngen:    
    children = cross_over(population = population, mut_rate=mut_rate, cross_rate=cross_rate, tsize=tsize, elitism=elitism, score=score)
    old_score = score
    score = []
    for z in range(popsize):
      score.append (fit_fun(children[z]))
    
    score = np.array(score)
    worst =  score.argsort()[0:(elitism)]

    children = np.array(children)
    prev_pop = population
    population = np.concatenate((children, np.array(prev_pop)[best])) 
    score = np.concatenate((score,old_score[best]))    
    population = np.delete(population, worst, axis=0) 
    score =  np.delete(score, worst)   
    best =  score.argsort()[-elitism:][::-1]  
    pop_average.append (np.mean(score))
    best_average.append (np.mean(score[best]))    
    population = population.tolist()
    
    gen+=1
    if (verbose == True):
      print('Generation:', gen)
      print('Population average:', pop_average[gen])
      print('Best:', best_average[gen])
    
  return (population, score, best, pop_average, best_average)
