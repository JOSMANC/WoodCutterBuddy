import cPickle as  pickle
import requests
import pandas as pd
import numpy as np
import pygeocoder as pgeo
from datetime import date

class RealtorBuddy(object):
'''
The prototype class for predicting home prices using the optimally trainned model.

The class could be used interactively with a Flask app which would pipe in 
input about from the user into the model.  Information like the address,
taxes, lotsize, bedroom, and home characteristics are loaded into the model.
The model process the address to find the geolat and geolon to census block
group features.  Potentially useful information for various databases are queried
these are represented by dictionaries in the prototype.  The features are passed to 
the model and a price is evaluated for input home.
'''
    def  __init__(self):
        self.state = None
        self.county = None
        self.tract = None
        self.block_group = None
        self.geolat = None 
        self.geolon = None
        self.census_info = None
        self.school_scores = None
        return None

    def _clean_address(self, input):
        '''
        '''
        clean_address = [add.strip().lower().replace(' ','+') for add in input]
        return clean_address

    def compute_census_location(self, input):
        '''
        '''
        address = self._clean_address(input)
        census_query_middle_prefixs = ['street=','&city=','&state=','&zip=']
        census_query_prefix = 'http://geocoding.geo.census.gov/geocoder/geographies/address?'
        census_query_sufix = '&benchmark=Public_AR_Census2010&vintage=Census2010_Census2010&layers=14&format=json'
        census_query_middle = [None]*8
        census_query_middle[::2] = census_query_middle_prefixs
        census_query_middle[1::2] = address
        census_query_middle = ''.join(census_query_middle)

        census_query  = census_query_prefix+census_query_middle+census_query_sufix
        print census_query
        response = requests.get(census_query)
        fips = response.json()['result']['addressMatches'][0]['geographies']['Census Blocks'][0]['GEOID']
        self.break_fips(fips)
        return None

    def compute_geolat_geolon(self, address):
        '''
        '''
        self.geolat, self.geolon = pgeo.Geocoder.geocode(address).coordinates
        return None

    def break_fips(self, fips):
        '''
        '''
        self.state = fips[0:2]
        self.county = fips[2:5]
        self.tract = fips[5:11]
        self.block_group = fips[11:12]
        return None

    def compute_census_area_features(self):
        '''
        '''
        df_cs = pd.read_pickle('census_statistics.pkl')
        census_info = df_cs[df_cs['countytractblock_group']==(self.county+self.tract+self.block_group)]
        return census_info

    def compute_school_features(self):
        '''
        '''
        school_scorer_nn = pickle.load(file('school_scorer_nn.pkl', 'rb'))
        school_scores = school_scorer_nn.predict([self.geolat, self.geolon])
        _, school_distance = school_scorer_nn.kneighbors([self.geolat, self.geolon])
        return school_scores, school_distance

    def compute_commerical_features(self):
        '''
        '''
        commerical_buisness_dict = pickle.load(file('commerical_buisness_dict.pkl', 'rb')) 
        commerical_buisness_nn = pickle.load(file('commerical_nn.pkl', 'rb'))
        local_commerical_buisnesses = np.sum([commerical_buisness_dict[b] for b in commerical_buisness_nn.kneighbors([self.geolat, self.geolon],return_distance=False)[0]])
        return local_commerical_buisnesses

    def compute_census_home_features(self):
        '''
        '''
        df_cc = pickle.load(file('census_comp_info.pkl', 'rb'))
        cenus_area_comps = df_cc[df_cc['countytractblock_group']==(self.county+self.tract+self.block_group)].values[0].tolist()[1:]
        '''
        alternative posgress query 

        '''
        return cenus_area_comps

    def compute_timepoint(self):
        '''
        '''
        days_from_zero = (date.today()-date(2009,05,12)).days
        return days_from_zero

    def predict_price(self, features_vector, characteristics_vector):
        '''
        '''
        rfr_for_predict = pickle.load(file('rfr_predict.pkl', 'rb'))
        pca_for_predict = pickle.load(file('pca_for_predict.pkl', 'rb'))
        pca_vector = pca_for_predict.transform(characteristics_vector)
        predictor.predict(data)
        data_components = data['']



if __name__ == "__main__":
    rtr = RemoveTheRealtor()
    address = []
    manditory_feature_labels = []
    address = []
    #manditory_feature_labels = ['livingarea', 'approxlotsqft', 'taxes']
    manditory_feature = [1596, 6954, 702.00]
    ''' general feature categories
    'propertydescription', 'roofing', 'construction', 'unitstyle',
    'exteriorfeatures', 'fencing', 'features', 'accessibilityfeat',
    'communityfeatures', 'kitchenfeatures', 'spa', 'landscaping',
    'basementdescription', 'poolprivate', 'masterbedroom',
    'masterbathroom', 'otherrooms', 'laundry'    
    '''
    characteristics_vector = \
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
           0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    rtr.compute_geolat_geolon(' '.join(address))
    rtr.compute_census_location(address)
    cenus_area_comps = rtr.compute_census_home_features()
    days_from_zero = rtr.compute_timepoint()
    feature_vector = manditory_feature + cenus_area_comps + [days_from_zero]
    local_commerical_buisnesses = rtr.compute_commerical_features()
    compute_census_area_features = rtr.compute_census_area_features()
    school_scores, school_distance = rtr.compute_school_features()
    price_prediction = rtr.predict_price(feature_vector, characteristics_vector)
    return price_prediction

