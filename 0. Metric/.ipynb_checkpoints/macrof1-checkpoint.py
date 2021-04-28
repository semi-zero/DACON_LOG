from sklearn import metrics

def macro_f1(answer_df, submission_df):
    submission_df = submission_df[submission_df['id'].isin(answer_df['id'])]
    submission_df.index = range(submission_df.shape[0])
    
    true = answer_df['level']
    pred = submission_df['level']
    
    score = metrics.f1_score(y_true=true, y_pred=pred, average='macro')
    
    return score