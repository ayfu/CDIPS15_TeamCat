all_files = {}
for afile in allfiles:
    key = afile.split('\\')[-1].split('.')[0]
    all_files[key] = pd.read_csv(afile, header = 0)
all_files['train_set'] = pd.read_csv('../competition_data/train_set.csv', header = 0, parse_dates = ['quote_date'])
all_files['test_set'] = pd.read_csv('../competition_data/test_set.csv', header = 0, parse_dates = ['quote_date'])

train[all_files['bill_of_materials'].columns]
dropcol = ['component_id_7','quantity_7','component_id_8','quantity_8']
test[all_files['bill_of_materials'].drop(dropcol, axis = 1).columns]



train[all_files['tube'].columns]
test[all_files['tube'].columns]



weightlook = train.columns[train.columns.str.contains('weight')]
weightlookTest = train.columns[train.columns.str.contains('weight')]
train[weightlook]
test[weightlookTest]


train[['year','month','day']]
test[['year','month','day']]


train[['end_a','end_x']]
test[['end_a','end_x']]

endformtype = train.columns[train.columns.str.contains('end_form')]
endformtypeTest = test.columns[test.columns.str.contains('end_form')]
train[endformtype]
test[endformtypeTest]


contype = train.columns[train.columns.str.contains('conn')]
contypeTest = test.columns[test.columns.str.contains('conn')]
train[contype]
test[contypeTest]


comptype = train.columns[train.columns.str.contains('component_type')]
comptypeTest = test.columns[test.columns.str.contains('component_type')]
train[comptype]
test[comptypeTest]


restfile = []
for x in rest:
    restfile += [os.path.join('..','competition_data',x)]
rest_files = {}
for r in restfile:
    key = r.split('\\')[-1].split('.')[0]
    rest_files[key] = pd.read_csv(r ,header=0)
train[list(rest_files['specs'].columns)]
test[list(rest_files['specs'].columns)]



componName = train.columns[train.columns.str.contains('component_id')]
componNameTest = test.columns[test.columns.str.contains('component_id')]
train[componName]
test[componName]
