import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import random
import yfinance as yf
import pandas as pd


#this algorithm/model is created based on the initial idea/assumption that at every point in time (n) point a stock has a 50/50 chance of increasing or decreasing by 1%.
#if the prior model is created mathematically, you will end up with a problem which is computationally inpheasible (2^n). However, you will also find that the confidence intervals at each point in time,
#become increasingly "precise" or "certain" as time increases due to the expanding nature of the dataset. The PlotIt() function creates the theorized model above, with endtime, beginning price, and average increase/decrease parameterized.
#LongRun() simulates n passes of randomized "1%" increases/decreases over a parameterised period of time

#im fairly confident that the equations arrived at in the plotit function can be generalized mathematically. That will be completed at a later date.

#avgincrease=1.011267632
#avgdecrease=0.9870758867
#pricetime0=52.58
#increaseprobability=0.5454545455
#decreaseprobability=1-increaseprobability
####    7 months of training with a 30 day prediction time resulted in a $0.01 discrepency in predicted and actual of MSFT on the exact day being predicted for
####    this is probably a coincidence but I decided to take note since nothing was tuned for this result, i simply picked the training parameters and dates somewhat randomly.
####    this occured with the first implementation of bear/bull glue in the LongRun function using 1000 passes. I will upload to git on this commit to track further successes

#these models are better for long-term predictions whereby they do not take into consideration trends. Trends can be factored in using another algorithm ive been thinking of where we can demark changes in
#the stocks behavior by looking at abnormal changes in volatility through hypothesis testing, then using this demarkation to determine the training data's timeframe, or even take a time based weighted average through multiple demarkations

trainingcutoff='2021-01-20'


def GetData():
    print('why are u running')
    data = yf.download(tickers="GME",start='2021-01-01',end=trainingcutoff)#,interval="1m")
    df=pd.DataFrame(data=data)
    df['pctchange']=(df.Close-df.Open)/df.Open
    pctchange=df['pctchange'].to_list()
    interval=st.norm.interval(confidence=0.99999, loc=np.mean(pctchange), scale=st.sem(pctchange))
    print(interval)
    df['abnormal?']=(df['pctchange']<interval[0]) | (df['pctchange']>interval[1])
    print(df)
    lastofem=df['Close'].to_list()
    lastofem=lastofem[len(lastofem)-1]
    df2=df.loc[df['pctchange']<0]
    neglist=df2['pctchange'].to_list()
    df2=df.loc[df['pctchange']>=0]
    poslist=df2['pctchange'].to_list()
    averageincrease=np.mean(poslist)
    averagedecrease=np.mean(neglist)
    increaserate=len(poslist)/len(pctchange)
    decreaserate=1-increaserate
    print('it ran')
    gluelist=[]
    for x in pctchange:
        if x <0:
            gluelist.append('Bear')
        else:
            gluelist.append('Bull')
    counter=0
    BearGlue=[]
    BullGlue=[]
    BearStickiness=2
    BullStickiness=2
    for x in gluelist:
        if counter==0:
            counter=counter+1
            continue
        if gluelist[counter-1]==x:
            if x=='Bear':
                BearStickiness=BearStickiness+1
            if x=='Bull':
                BullStickiness=BullStickiness+1
        else:
            if x=='Bear':
                BullGlue.append(BullStickiness)
                BullStickiness=2
            if x=='Bull':
                BearGlue.append(BearStickiness)
                BearStickiness=2
        counter=counter+1
    print(BearGlue,BullGlue)
    BearGlueDuration=round(np.mean(BearGlue),0)
    BearGlueContagion=len(BearGlue)/len(neglist)
    BullGlueDuration=round(np.mean(BullGlue),0)
    BullGlueContagion=len(BullGlue)/len(poslist)

    return lastofem,averageincrease+1,averagedecrease+1,increaserate,decreaserate,BearGlueDuration,BearGlueContagion,BullGlueDuration,BullGlueContagion



pricetime0,avgincrease,avgdecrease,increaseprobability,decreaseprobability,BearGlueDuration,BearGlueContagion,BullGlueDuration,BullGlueContagion=GetData()

#print(GetData())

def GetDataPoints(datapoints):
    templist=[]
    for price in datapoints:
        templist.extend([price*avgincrease,price*avgdecrease])
    return templist

def GetDataPointsAtTime(time):
    datapoints=[pricetime0]
    for int in range(time):
        datapoints=GetDataPoints(datapoints)
        #print(datapoints)
    return datapoints

def GetConfidenceInterval(time):
    datapoints=GetDataPointsAtTime(time)
    if time!=0:
        interval=st.norm.interval(confidence=0.99, loc=np.mean(datapoints), scale=st.sem(datapoints))
        return time,max(interval),min(interval),max(datapoints),np.mean(datapoints),min(datapoints)
    else:
        return time,datapoints[0],datapoints[0],datapoints[0],datapoints[0],datapoints[0]

def ConstructDataset(endtime):
    xlist=[]
    maxintlist=[]
    minintlist=[]
    maxdatalist=[]
    meandatalist=[]
    mindatalist=[]
    for time in range(endtime):
        x,maxint,minint,maxdata,meandata,mindata=GetConfidenceInterval(time)
        xlist.append(x)
        maxintlist.append(maxint)
        minintlist.append(minint)
        maxdatalist.append(maxdata)
        meandatalist.append(meandata)
        mindatalist.append(mindata)
    return [xlist,maxintlist,minintlist,maxdatalist,meandatalist,mindatalist]

def PlotIt(Endtime):
    dataset=ConstructDataset(Endtime)
    skip=True
    counter=0
    headers=['CI+','CI-','Max',"Mean",'Min']
    for x in dataset:
        if skip==True:
            skip=False
            continue
        else:
            plt.plot(dataset[0], x, label = headers[counter])
            counter=counter+1
    plt.legend()
    plt.show()

#PlotIt(20)

def LongRun(passes,endtime):
    print('getting to work...')
    lastitemlist=[]
    BearTimeout=0
    BearMode=False
    BullTimeout=0
    BullMode=False
    for intmain in range(passes):
        
        if random.uniform(0,1) >=increaseprobability:
            templist=[pricetime0*avgincrease]
        else:
            templist=[pricetime0*avgdecrease]
        
        for int in range(1,endtime):

            if int>=BearTimeout and BearMode==True:
                BearMode=False
                templist.append(templist[len(templist)-1]*avgincrease)
                continue 

            if (templist[int-1]<0 and random.uniform(0,1)>BearGlueContagion) and BearMode==False:
                BearTimeout=intmain+BearGlueDuration+1
                BearMode=True
                templist.append(templist[len(templist)-1]*avgdecrease)
                continue
                
            if int>=BullTimeout and BullMode==True:
                BullMode=False
                templist.append(templist[len(templist)-1]*avgdecrease)
                continue 

            if (templist[int-1]>=0 and random.uniform(0,1)>BullGlueContagion) and BullMode==False:
                BullTimeout=intmain+BullGlueDuration+1
                BullMode=True
                templist.append(templist[len(templist)-1]*avgincrease)
                continue

            if random.uniform(0,1)>=increaseprobability:
                change=avgincrease
            else:
                change=avgdecrease
            newval=templist[len(templist)-1]*change
            templist.append(newval)
        lastitemlist.append(newval)
        plt.plot(range(endtime),templist,label='intmain')
    plt.show()
    plt.hist(lastitemlist)
    plt.show()
    print(np.mean(lastitemlist))

LongRun(1000,30)