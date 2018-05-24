"""
    Capsule8 DataScience Challenge
    Code Developed by Harshith Guru Prasad
    USC | MS in CS | Data Science and Machine Learning Enthusiast
"""

#import the required packages/libraries 
import sqlite3
from sqlite3 import Error
import json
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVR

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None

def retrieve_schema(conn, table_name):
    
    headers = conn.execute("PRAGMA table_info("+table_name+");")
    Col_Label = []
    for header in headers:
        Col_Label.append(header)   
    return Col_Label

def retrieve_tuples(conn, table_name):
    
    pointer = conn.execute("SELECT * FROM "+table_name)
    Tuples = []
    for tup in pointer:
        Tuples.append(tup)   
    
    if len(Tuples)>0:
        return Tuples

def plot_graph(X,Y,color, ylabel):
    plt.title('Time Series Chart')
    plt.xlabel('time')
    plt.ylabel(ylabel)
    plt.plot(X,Y,color=color)
    plt.show()

def plot_histogram(data):
    plt.hist(data, bins = 10)
    
def main():
    
    #creating a connection with the database
    Conn = create_connection('sqlite.db')
    
    Table_exec = retrieve_tuples(Conn, 'exec')
    
    #exploring the table exec
    """
    headers = retrieve_schema(Conn, 'exec')
    for h in headers:
        print(h)
    """
    
    Time_Stamp = []  
    Depth = []
    Rate_1s = []
    Rate_5s = []
    
    for tup in Table_exec:
        Time_Stamp.append(tup[0])
        Depth.append(tup[-2])
        rate = (tup[-1])
        d = json.loads(rate)
        Rate_1s.append(d["1"])
        Rate_5s.append(d["5"])
    
    #plotting time series graphs
    plot_graph(Time_Stamp, Depth, color='blue', ylabel='depth')
    plot_graph(Time_Stamp, Rate_1s, color='red', ylabel='rate_1s')
    plot_graph(Time_Stamp, Rate_5s, color='green', ylabel='rate_5s')
    
    #plotting hitograms
    plot_histogram(Depth)
    plot_histogram(Rate_1s)
    plot_histogram(Rate_5s)
    
    #exploring the table tcplife
    Table_tcpLife = retrieve_tuples(Conn, 'tcplife')
    """
    headers = retrieve_schema(Conn, 'tcplife')
    for h in headers:
        print(h)
    """
    Time_Stamp1 = []
    Rx = []
    Tx = []
    Dur = []
    Lport = []
    Rport = []
    
    for tup in Table_tcpLife:
        Time_Stamp1.append(tup[0])
        Rx.append(tup[4])
        Tx.append(tup[5])
        Dur.append(tup[6])
        Lport.append(tup[2])
        Rport.append(tup[3])
    
    #Time series charts for tcplife     
    plot_graph(Time_Stamp1, Rx, color='blue', ylabel='RX')
    plot_graph(Time_Stamp1, Tx, color='red', ylabel='TX')
    plot_graph(Time_Stamp1, Dur, color='green', ylabel='DUR')
    plot_graph(Time_Stamp1, Lport, color='orange', ylabel='Lport')
    plot_graph(Time_Stamp1, Rport, color='pink', ylabel='Rport')
    
    # plotting histograms
    plot_histogram(Rx)
    plot_histogram(Tx)
    plot_histogram(Dur)
    plot_histogram(Lport)
    plot_histogram(Rport)
    
    """Descriptive Statistics"""
    import statistics
    from scipy.stats import mode as M
    mean = statistics.mean(Dur)
    median = statistics.median(Dur)
    mode = M(Dur)  
    
    from collections import defaultdict
    value_count = defaultdict(int)
    for d in Dur:
        value_count[d]+=1
    
    t=[]
    for w in sorted(value_count, key=value_count.get, reverse=True):
        t.append([w, value_count.get(w)])
    T = t[0]
    """
    Mean = 261.219
    Median = 144.649
    Mode = 

    """
    
    feat = []
    for t in Time_Stamp1:
        feat.append([t])
    
    """      
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(feat, Lport)
    plt.plot(feat, Lport)
    plt.plot(feat, lin_reg.predict(feat), color='red')
   """
    
    # find average difference between the timestamps
    import math
    sum=0
    count=0
    Diff = []
    for t in range(0,len(Time_Stamp1)-1):
        diff = math.fabs(Time_Stamp1[t] - Time_Stamp1[t+1])
        Diff.append(diff)
        sum+=diff
        count+=1
    print(sum)
    avg_diff = sum/count
        
    test = []
    for d in Dur:
        test.append(d)
    for l in Lport:
        test.append(l)
    
    new_test = []
    x=Time_Stamp1[-1]
    for i in range(0,25):
        new_test.append([x+(i+1)*avg_diff])
    
    #y_pred = lin_reg.predict(new_test)
    
    svr_rbf = SVR(kernel='rbf', C=10e3, gamma=0.5)
    #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    #svr_lin = SVR(kernel='linear', C=1e3)
    #svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    ypred = svr_rbf.fit(feat, Dur).predict(new_test)
    #lw = 2
       
    #y_lin = svr_lin.fit(X, y).predict(X)
    #y_poly = svr_poly.fit(feat, DUR).predict(feat)
    plt.plot(feat, Dur)
    plt.plot(feat, svr_rbf.predict(feat), color='red', lw=1, label='RBF model')
    
    

    svr_rbf_1 = SVR(kernel='rbf', C=100e3, gamma=0.9)
    ypred_1 = svr_rbf_1.fit(feat, Lport).predict(new_test)
    plt.plot(feat, Lport)
    plt.plot(feat, svr_rbf_1.predict(feat), color='red', lw=1, label='RBF model')

    
if __name__ == '__main__':
    main()    
    
"""  
conn.close()
"""

