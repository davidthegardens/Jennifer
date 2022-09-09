import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import random
import yfinance as yf
import pandas as pd
import datetime

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

### ADD sensitivity toggling based on volatility, pltr works well with 1.1, MSFT with 5
def ConvertToDateObj(DateString):
    lillist=DateString.split('-')
    year,month,day=int(lillist[0]),int(lillist[1]),int(lillist[2])
    return datetime.date(year=year,month=month,day=day)

def AbnormalSplit(SepList):
    SepLenList=[]
    counter=0
    threshold=3
    for i in SepList:
        if i==True and counter>threshold:
            SepLenList.append(counter)
            counter=1
            #print('break')
        else:
            counter=counter+1
            #print(counter)
    if sum(SepLenList)!=len(SepList):
        SepLenList.append(counter)
    return SepLenList

#print(AbnormalSplit(TestAbnormalityList))
#print(sum(TimeWeighting(5)))

def Splitter(TargetList,SepLenList,df):
    SplittedList=[]
    counter=0
    dflist=[]
    for int in SepLenList:
        templist=[]
        for i in range(1,int+1):
            templist.append(TargetList[counter+(i-1)])
        SplittedList.append(templist)
        counter=counter+len(templist)
    
    for listy in SplittedList:
        tempdf=df[df['Date'].isin(listy)]
        dflist.append(tempdf)
    return dflist

#predictiondate='2022-09-06'
#predictionperiod=30
#trainingperiod=600
#ticker="GE"
#endtrainingperiod=ConvertToDateObj(predictiondate)-datetime.timedelta(days=predictionperiod)
#starttrainingperiod=endtrainingperiod-datetime.timedelta(days=trainingperiod)

def TimeWeighting(n,TimeWeightSensitivity):
    TimeWeighting=[]
    for int in range(1,n+1):
        TimeWeighting.append(1/(TimeWeightSensitivity**int))
    Remainder=(1-sum(TimeWeighting))/n
    templist=[]
    for i in TimeWeighting:
        templist.append(i+Remainder)
    templist.sort()
    return templist

def TimeWeightedData(listofdfs,TimeWeightSensitivity):
    avginclist=[]
    avgdeclist=[]
    incratelist=[]
    decratelist=[]
    for df in listofdfs:
        pctchange=df['pctchange'].to_list()
        df2=df.loc[df['pctchange']<0]
        neglist=df2['pctchange'].to_list()
        df2=df.loc[df['pctchange']>=0]
        poslist=df2['pctchange'].to_list()
        avginclist.append(np.mean(poslist))
        avgdeclist.append(np.mean(neglist))
        incratelist.append((len(poslist)/len(pctchange)))
        decratelist.append((1-(len(poslist)/len(pctchange))))
    keydata=[avginclist,avgdeclist,incratelist,decratelist]
    keyvalues=[]
    Weighting=TimeWeighting(len(avginclist),TimeWeightSensitivity)
    for listy in keydata:
        counter=0
        templist=[]
        for floater in listy:
            if np.isnan(floater) and counter!=0:
                floater=priorfloater
            elif np.isnan(floater) and counter==0:
                floater=listy[counter+1]
            priorfloater=floater
            templist.append(floater*Weighting[counter])
            #print(floater,Weighting[counter])
            counter=counter+1
        
        #print(templist)
        entry=sum(templist)
        #print(entry)
        keyvalues.append(entry)
    return keyvalues[0],keyvalues[1],keyvalues[2],keyvalues[3]



def GetData(TimeWeightSensitivity,predictiondate,predictionperiod,ticker):
    trainingperiod=600
    endtrainingperiod=ConvertToDateObj(predictiondate)-datetime.timedelta(days=predictionperiod)
    starttrainingperiod=endtrainingperiod-datetime.timedelta(days=trainingperiod)
    print('why are u running')
    data = yf.download(tickers=ticker,start=str(starttrainingperiod),end=str(ConvertToDateObj(predictiondate)+datetime.timedelta(days=1)))#,interval="1m")
    df=pd.DataFrame(data=data)
    df=df.dropna()
    actualsdf=df.tail(predictionperiod)
    df.drop(df.tail(predictionperiod).index,inplace = True)
    df['pctchange']=(df.Close-df.Open)/df.Open
    pctchange=df['pctchange'].to_list()
    interval=st.norm.interval(confidence=0.999999999, loc=np.mean(pctchange), scale=st.sem(pctchange))
    #print(interval)
    df['abnormal?']=(df['pctchange']<interval[0]) | (df['pctchange']>interval[1])
    abnormallist=df['abnormal?'].to_list()
    fulldf=df
    lastofem=fulldf['Close'].to_list()
    lastofem=lastofem[len(lastofem)-1]
    fulldf['Date']=df.index
    datelist=fulldf['Date'].to_list()
    averageincrease,averagedecrease,increaserate,decreaserate=TimeWeightedData(Splitter(datelist,AbnormalSplit(abnormallist),fulldf),TimeWeightSensitivity)
    lastofemdate=datelist[len(datelist)-1]
    lastofemdate=ConvertToDateObj(str(lastofemdate).split(' ')[0])
    df2=fulldf.loc[fulldf['pctchange']<0]
    neglist=df2['pctchange'].to_list()
    df2=fulldf.loc[fulldf['pctchange']>=0]
    poslist=df2['pctchange'].to_list()
    
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
    #print(BearGlue,BullGlue)
    BearGlueDuration=round(np.mean(BearGlue),0)
    BearGlueContagion=len(BearGlue)/len(neglist)
    BullGlueDuration=round(np.mean(BullGlue),0)
    BullGlueContagion=len(BullGlue)/len(poslist)
    print('it ran')
    return lastofem,averageincrease+1,averagedecrease+1,increaserate,decreaserate,BearGlueDuration,BearGlueContagion,BullGlueDuration,BullGlueContagion,lastofemdate,actualsdf,datelist,abnormallist,df



#pricetime0,avgincrease,avgdecrease,increaseprobability,decreaseprobability,BearGlueDuration,BearGlueContagion,BullGlueDuration,BullGlueContagion,lastofemdate,actualsdf,datelist,abnormallist,df=GetData()
#print(BullGlueContagion,BearGlueContagion)
#print(GetData())

def GetDataPoints(datapoints,avgincrease,avgdecrease):
    templist=[]
    for price in datapoints:
        templist.extend([price*avgincrease,price*avgdecrease])
    return templist

def GetDataPointsAtTime(time,pricetime0):
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

def LongRun(passes,endtime,plot,pricetime0,avgincrease,avgdecrease,increaseprobability,decreaseprobability,BearGlueDuration,BearGlueContagion,BullGlueDuration,BullGlueContagion,lastofemdate,actualsdf,datelist,abnormallist,df,predictiondate,predictionperiod):
    print('getting to work...')
    df=actualsdf
    df['Date']=df.index
    xaxis=df['Date'].to_list()

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
        if plot==True:
            plt.plot(xaxis,templist,label='intmain')
    if plot==True:
        plt.show()
        plt.hist(lastitemlist)
        plt.show()
    df=df.tail(1)
    High=df['High'].to_list()[0]
    Low=df['Low'].to_list()[0]
    Result=np.mean(lastitemlist)
    if Result>=Low and Result<=High:
        Accuracy=1
    elif Result<Low:
        Miss=(Low-Result)/Low
        Accuracy=1-Miss
    elif Result>High:
        Miss=(Result-High)/High
        Accuracy=1-Miss
    if plot==True:
        print(Result,' on ',predictiondate,' using ',predictionperiod,' days blinded with last known price being ',pricetime0, 'the price ranged from ',Low,' to ',High,'. Accuracy is ',Accuracy)
    return Accuracy

#LongRun(1000,predictionperiod,False)

def OptimalSensitivity(Ticker,PredictionPeriod,TestDepth):
    TimeWeightSensitivity=range(1,30)
    TestResults=[]
    for Sensitivity in TimeWeightSensitivity:
        TempTestResults=[]
        for i in range(1,TestDepth):

            Predictiondate=str(datetime.date.today()-datetime.timedelta(days=random.choice(range(7,1000))))

            #Predictiondate=Predictiondate.strftime("%d/%m/%Y")
            pricetime0,avgincrease,avgdecrease,increaseprobability,decreaseprobability,BearGlueDuration,BearGlueContagion,BullGlueDuration,BullGlueContagion,lastofemdate,actualsdf,datelist,abnormallist,df=GetData(Sensitivity,Predictiondate,PredictionPeriod,Ticker)
            TempTestResults.append(LongRun(100,PredictionPeriod,False,pricetime0,avgincrease,avgdecrease,increaseprobability,decreaseprobability,BearGlueDuration,BearGlueContagion,BullGlueDuration,BullGlueContagion,lastofemdate,actualsdf,datelist,abnormallist,df,Predictiondate,PredictionPeriod))
        TestResults.append(np.mean(TempTestResults))
    return TestResults,TimeWeightSensitivity
    
#print(TimeWeighting(5))

print(OptimalSensitivity('MSFT',30,10))

#print(TimeWeightedData(Splitter(datelist,AbnormalSplit(abnormallist),df)))