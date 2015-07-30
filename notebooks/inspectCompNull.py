# Create a dictionary of Series for each component 1,2,3,4,5,6,7 features
# Each Series shows the percentage of null values in each corresponding column in traintest

traintest = pd.read_csv('traintest.csv', header = 0)
comp1 = traintest.iloc[:,10:65]
comp2 = traintest.iloc[:,65:121]
comp3 = traintest.iloc[:,121:186]
comp4 = traintest.iloc[:,186:242]
comp5 = traintest.iloc[:,242:283]
comp6 = traintest.iloc[:,283:312]
comp7 = traintest.iloc[:,312:316]

dfcomp = [comp1,comp2,comp3,comp4,comp5,comp6,comp7]
component = ['comp'+str(i) for i in range(1,8)]
compdict = {}
for i in range(len(component)):
    compdict[component[i]] = dfcomp[i]



df = {}
for comp in component:
    tempdict = {}
    for x in compdict[comp].columns:
        tempdict[x] = np.mean(pd.isnull(compdict[comp][x]))
    tempdict = pd.Series(tempdict)
    df[comp] = tempdict
