import uproot
import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt
import tensorflow            as tf
import os
import networkx as nx

from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from spektral.data.dataset import Dataset, Graph




############################### CLASS for data preparation ###############################

class DataPreparation:

    ############################### DATA PREPARATION ###############################

    def __init__(self,
                 file1_path,
                 treeS_name = "newtreeS",
                 treeB_name = "newtreeB"):
        """
        CONSTRUCTOR to initialize the DataPreparation object.
        Parameters:
        - file1_path (str): Path to the ROOT file containing data.
        - treeS_name (str): Name of the signal tree. Default is "newTreeS".
        - treeB_name (str): Name of the background tree. Default is "newTreeB".
        This constructor sets the file path and tree names for the DataPreparation object.
        """

        # Initialize the DataPreparation OBJECT with the ROOT file path and tree names
        self.file1_path = file1_path
        self.treeS_name = treeS_name
        self.treeB_name = treeB_name

    ############################### LOAD DATA ###############################

    def load_data(self):  # Loading data
        print("------------------------------------------------------------uproot.open--------------------")
        file1 = uproot.open(self.file1_path)

        print("------------------------------------------------------------extract--------------------")
        # Extract signal and background trees
        treeS = file1[self.treeS_name] #signal
        treeB = file1[self.treeB_name] #background

        # Print what imported
        print(treeS)
        print(treeB)
        print("\n")

        treeS.show()
        print("\n")
        treeB.show()
        print("\n")

        print("------------------------------------------------------------.arrays--------------------")
        # Load data from Trees into Pandas DataFrames (df)
        print("--------------------------------------------------------------------------------df_signal--------------------")
        self.df_signal     = treeS.arrays(library = "pd", entry_stop=45000)
        #self.df_signal     = treeS.arrays(library = "pd")
        print(self.df_signal)
        
        print("--------------------------------------------------------------------------------df_background--------------------")
        #self.df_background = treeB.arrays(library = "pd", entry_stop=1000000)
        self.df_background = treeB.arrays(library = "pd", entry_stop=45000)
        print(self.df_background)

    ############################### PREPARE DATA ###############################

    def prepare_data(self,
                     model_type):  # This function is designed to prepare data for the training and evaluation of a classification model.
        # Select features
        print("------------------------------------------------------------X_signal--------------------")
        X_signal = self.df_signal[["massK0S",
                                   "tImpParBach",
                                   "tImpParV0",
                                   "CtK0S",
                                   "cosPAK0S",
                                   "nSigmapr",
                                   "dcaV0"]]
        print(X_signal)

        print("------------------------------------------------------------X_background--------------------")
        X_background = self.df_background[["massK0S",
                                           "tImpParBach",
                                           "tImpParV0",
                                           "CtK0S",
                                           "cosPAK0S",
                                           "nSigmapr",
                                           "dcaV0"]]
        print(X_background)

        print("------------------------------------------------------------X_feature_names--------------------")
        self.feature_names = ["massK0S",
                              "tImpParBach",
                              "tImpParV0",
                              "CtK0S",
                              "cosPAK0S",
                              "nSigmapr",
                              "dcaV0"]

        ############################### Start Normalisation ###############################
        print("------------------------------------------------------------Normalization--------------------")
        print("--------------------------------------------------------------------------------Take the max_signal--------------------")
        # Normalize SIGNAL data by dividing by the maximum value of each variable
        max_massK0S_signal     = X_signal["massK0S"].max()
        max_tImpParBach_signal = X_signal["tImpParBach"].max()
        max_tImpParV0_signal   = X_signal["tImpParV0"].max()
        max_CtK0S_signal       = X_signal["CtK0S"].max()
        max_cosPAK0S_signal    = X_signal["cosPAK0S"].max()
        max_nSigmapr_signal    = X_signal["nSigmapr"].max()
        max_dcaV0_signal       = X_signal["dcaV0"].max()

        X_signal_normalized = pd.DataFrame()

        print("--------------------------------------------------------------------------------Divide_signal--------------------")
        X_signal_normalized["massK0S"]     = X_signal["massK0S"]     / max_massK0S_signal
        X_signal_normalized["tImpParBach"] = X_signal["tImpParBach"] / max_tImpParBach_signal
        X_signal_normalized["tImpParV0"]   = X_signal["tImpParV0"]   / max_tImpParV0_signal
        X_signal_normalized["CtK0S"]       = X_signal["CtK0S"]       / max_CtK0S_signal
        X_signal_normalized["cosPAK0S"]    = X_signal["cosPAK0S"]    / max_cosPAK0S_signal
        X_signal_normalized["nSigmapr"]    = X_signal["nSigmapr"]    / max_nSigmapr_signal
        X_signal_normalized["dcaV0"]       = X_signal["dcaV0"]       / max_dcaV0_signal

        print("--------------------------------------------------------------------------------X_signal_normalized--------------------")
        print(X_signal_normalized)

        print("--------------------------------------------------------------------------------Take the max_background--------------------")
        # Normalize BACKGROUND data by dividing by the maximum value of each variable
        max_massK0S_background     = X_background["massK0S"].max()
        max_tImpParBach_background = X_background["tImpParBach"].max()
        max_tImpParV0_background   = X_background["tImpParV0"].max()
        max_CtK0S_background       = X_background["CtK0S"].max()
        max_cosPAK0S_background    = X_background["cosPAK0S"].max()
        max_nSigmapr_background    = X_background["nSigmapr"].max()
        max_dcaV0_background       = X_background["dcaV0"].max()

        X_background_normalized = pd.DataFrame()

        print("--------------------------------------------------------------------------------Divide_background--------------------")
        X_background_normalized["massK0S"]     = X_background["massK0S"]     / max_massK0S_background
        X_background_normalized["tImpParBach"] = X_background["tImpParBach"] / max_tImpParBach_background
        X_background_normalized["tImpParV0"]   = X_background["tImpParV0"]   / max_tImpParV0_background
        X_background_normalized["CtK0S"]       = X_background["CtK0S"]       / max_CtK0S_background
        X_background_normalized["cosPAK0S"]    = X_background["cosPAK0S"]    / max_cosPAK0S_background
        X_background_normalized["nSigmapr"]    = X_background["nSigmapr"]    / max_nSigmapr_background
        X_background_normalized["dcaV0"]       = X_background["dcaV0"]       / max_dcaV0_background

        print("--------------------------------------------------------------------------------X_background_normalized--------------------")
        print(X_background_normalized)

        ############################### End Normalisation ###############################

        print("--------------------------------------------------------------------------------X_background_normalized[:943645]--------------------")
        print(X_background_normalized[:943645])


        print("------------------------------------------------------------Concatenation--------------------")
        # Concatenate normalized DataFrames
        X = pd.concat([X_signal_normalized,
                        X_background_normalized[:943645]])

        print("--------------------------------------------------------------------------------X--------------------")
        print(X)

        print("------------------------------------------------------------Add target--------------------")
        
        # y = np.concatenate([np.ones(len(X_signal_normalized)),
        #                    np.zeros(943645)])
        # Add a 'target' column to distinguish signal (1) from background (0)
        if model_type == "GNN":
            y = np.concatenate([np.ones(len(X_signal_normalized)),
                                np.zeros(len(X_background_normalized[:943645]))])
            #X.insert(7,"isSignal",y)
        else:
            #definition of the labels array   
            y = np.concatenate([np.ones(len(X_signal_normalized)),
                                np.zeros(len(X_background_normalized[:943645]))])
        
        
        
        print("--------------------------------------------------------------------------------y--------------------")
        print(y)

        print("------------------------------------------------------------Split data--------------------")
        # Split data into training and test sets
        if model_type != "GNN":
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                                    y,
                                                                                    test_size    = 0.2,
                                                                                    random_state = 42,
                                                                                    shuffle      = True)
        
            # SPLITTA DATASET IN TRAIN E TEST test_size=0.2


            print("--------------------------------------------------------------------------------X_train--------------------")
            print(self.X_train)

            print("--------------------------------------------------------------------------------y_train--------------------")
            print(self.y_train) 

            print("--------------------------------------------------------------------------------X_test--------------------")
            print(self.X_test)

            print("--------------------------------------------------------------------------------y_test--------------------")
            print(self.y_test)

            ############################### Definition of category ###############################
            # print("------------------------------------------------------------Definition of category--------------------")
     
            # Separate data into two groups based on the absolute value of "eta" => categorisation
            self.X_train_cat1 = self.X_train[self.X_train['massK0S'].abs() > 0.498]
            self.X_train_cat2 = self.X_train[self.X_train['massK0S'].abs() <= 0.498]

            self.y_train_cat1 = self.y_train[self.X_train['massK0S'].abs() > 0.498]
            self.y_train_cat2 = self.y_train[self.X_train['massK0S'].abs() <= 0.498]

            self.X_test_cat1  = self.X_test[self.X_test['massK0S'].abs() > 0.498]
            self.X_test_cat2  = self.X_test[self.X_test['massK0S'].abs() <= 0.498]
        
            self.y_test_cat1  = self.y_test[self.X_test['massK0S'].abs() > 0.498]
            self.y_test_cat2  = self.y_test[self.X_test['massK0S'].abs() <= 0.498]

            # # Dropping 'eta' column
            # self.X_train      = self.X_train.drop(columns=['CtK0S'])
            # self.X_test       = self.X_test.drop(columns=['CtK0S'])

            # self.X_train_cat1 = self.X_train_cat1.drop(columns=['CtK0S'])
            # self.X_test_cat1  = self.X_test_cat1.drop(columns=['CtK0S'])

            # self.X_train_cat2 = self.X_train_cat2.drop(columns=['CtK0S'])
            # self.X_test_cat2  = self.X_test_cat2.drop(columns=['CtK0S'])
            self.n_samples = 0

        if model_type == "GNN":
            print("---------------------------------------------------------------------------GRAPH_PREPARATION--------------------")
            print("------------------------------------------------------------------Sample_preparation--------------------")
            sample_size = 3000
            self.n_samples = int(len(X)/sample_size)
            X_sample = list()
            y_sample = list()
            for i in range(self.n_samples):
                if i == len(X)/sample_size and len(X)%self.n_samples != 0:
                    X_sample.append(X[i*sample_size:(i*sample_size+len(X)%self.n_samples)])
                    y_sample.append(y[i*sample_size:(i*sample_size+len(X)%self.n_samples)])
                else:
                    X_sample.append(X[i*sample_size:(i+1)*sample_size])
                    y_sample.append(y[i*sample_size:(i+1)*sample_size])
            print("shape di X_sample: ", len(X_sample),
                  "shape di y_sample: ", len(y_sample))
            print("shape di X: ", X.shape,
                  "shape di y: ", y.shape)

            self.dataset = list()
            self.y_test = np.empty(shape = 0)
            for i in range(self.n_samples):
                print("preparation of the sample number: ", i, "/", self.n_samples)
                #Create the graph containing the data for the GNN training
                #splitting the samples into three parts: training, validation, testing
                X_train, node_features_test_pd, y_train, y_test = train_test_split(X_sample[i],
                                                                    y_sample[i],
                                                                    test_size    = 0.3,
                                                                    random_state = 42,
                                                                    shuffle      = True)
            
                node_features_train_pd, node_features_val_pd, y_train, y_val = train_test_split(X_train,
                                                                                                          y_train, 
                                                                                                          test_size    = 0.5,
                                                                                                          random_state = 42,
                                                                                                          shuffle      = True)
            
                #remove the last rows of the dataframe if they contain a different number of elements
                if len(node_features_train_pd) != len(node_features_test_pd):
                    difference = len(node_features_train_pd) - len(node_features_test_pd)
                    if difference > 0:                    
                        node_features_train_pd = node_features_train_pd.head(len(node_features_test_pd))
                        y_train = y_train[:-difference]
                    else:
                        node_features_test_pd = node_features_test_pd.head(len(node_features_train_pd))
                        y_test = y_test[:difference]
                self.y_test = np.append(self.y_test, y_test)

                if len(node_features_train_pd) != len(node_features_val_pd):
                    difference = len(node_features_train_pd) - len(node_features_val_pd)
                    print("difference: ", difference)
                    if difference > 0:
                        node_features_train_pd = node_features_train_pd.head(len(node_features_val_pd))
                        y_train = y_train[:-difference]
                    else:
                        print("prima della modifica y di val:" , len(y_val))
                        node_features_val_pd = node_features_val_pd.head(len(node_features_train_pd))
                        y_val = y_val[:difference]
                    
                    print("lunghezza di train ", len(node_features_train_pd))
                    print("lunghezza di val: ", len(node_features_val_pd))
                    print("lunghezza di test ", len(node_features_test_pd))
                    print("lunghezza di train labels ", len(y_train))
                    print("lunghezza di val labels: ", len(y_val))
                    print("lunghezza di test labels", len(y_test))
            
                print("---------------------------------------------------------------------Train_adjacency_matrix--------------------")
            
                #a_train, edges_train = build_graph(labels= self.y_train)
                a_train = build_graph(labels= y_train,
                                      dataframe=node_features_train_pd)
            
                print("---------------------------------------------------------------------Validation_adjacency_matrix--------------------")
            
                #a_val, edges_val = build_graph(labels=self.y_val)
                a_val = build_graph(labels=y_val,
                                    dataframe=node_features_val_pd)
            
                print("----------------------------------------------------------------------Test_adjacency_matrix--------------------")
            
                #a_test, edges_test = build_graph(labels= self.y_test)
                a_test = build_graph(labels= y_test,
                                     dataframe= node_features_test_pd)
            
                #remove the columns needed for classification
                #node_features_train_pd = node_features_train_pd.drop(columns=['isSignal', 'index'])
                #node_features_val_pd = node_features_val_pd.drop(columns=['isSignal', 'index'])
                #node_features_test_pd = node_features_test_pd.drop(columns=['isSignal', 'index'])

                print(node_features_train_pd)
                print(node_features_val_pd)
                print(node_features_test_pd)

                #transform pandas dataframes into numpy arrays
                node_features_train = node_features_train_pd.to_numpy()
                node_features_val = node_features_val_pd.to_numpy()
                node_features_test = node_features_test_pd.to_numpy()

                print("shape di X_train:", node_features_train.shape,
                "shape di y_train: ", y_train.shape,
                "shape di X_val:",node_features_val.shape,
                "shape di y_val: ", y_val.shape,
                "shape di X_test: ", node_features_test.shape,
                "shape di y_test: ", y_test.shape)
            
                dataset_train = GNNDataset(node_features=node_features_train,
                                           a_matrix=a_train,
                                           #edge_features= edges_train,
                                           labels= y_train)
                dataset_val = GNNDataset(node_features=node_features_val,
                                         a_matrix=a_val,
                                         #edge_features= edges_val,
                                         labels= y_val)
                dataset_test = GNNDataset(node_features=node_features_test,
                                          a_matrix= a_test,
                                          #edge_features= edges_test,
                                          labels= y_test)

                self.dataset.append([dataset_train, dataset_val, dataset_test])
                
                if i%10 == 0:
                    print("----------------------------------------------------------------------Graph_rep--------------------")
                    #visual representation of the trainig graph: each node is revelation
                    G = nx.from_scipy_sparse_array(a_train)
                    plt.figure(figsize=(10,6))
                    nx.draw_networkx(G, with_labels=True, node_color = 'lightblue', node_size = 20)
                    plt.title("Graph of the training sample data")
                    plt.axis('off')
                    if not os.path.exists("graph_rep"):
                        os.makedirs("graph_rep")
                    plt.savefig("graph_rep/dataset_sample.svg")

            #set to None the attribute not used in this model
            self.X_test, self.X_train, self.X_train_cat1, self.X_train_cat2, self.X_test_cat1, self.X_test_cat2 = None, None, None, None, None, None
            self.y_train = None
            
class GNNDataset(Dataset):
    def __init__(self, 
                 node_features,
                 a_matrix,
                 #edge_features,
                 labels):
        self.node_features = node_features
        self.labels = tf.convert_to_tensor(labels)
        #self.a_matrix = coo_matrix((labels.size, labels.size))
        #self.edge_features = edge_features
        self.a_matrix = a_matrix
        super().__init__()
    def read(self):
        graph = Graph(x=self.node_features, 
                      a= self.a_matrix, 
                      #e=self.edge_features, 
                      y=self.labels)
        return [graph]

def build_graph(labels, 
                dataframe):
    index = list()
    for i in range(len(dataframe)):
        index.append(i)
    dataframe.insert(7, "index", index)

    nonzero_rows = np.empty(shape=0)
    nonzero_col = np.empty(shape=0)
    
    #extract the signal from the dataframe
    #signal = dataframe[dataframe["isSignal"]==1]
    signal = dataframe[(dataframe["nSigmapr"]>-0.5) & (dataframe["nSigmapr"]<0.5)]
    signal_lenght = len(signal.index)
    signal_index = signal['index'].to_numpy()

    permutation = signal_index.tolist()
    for i in range(signal_lenght-1):
        permutation = permutation[-1:] + permutation[:-1]
        nonzero_col = np.append(nonzero_col, permutation)

    for i in range(signal_lenght-1):
        nonzero_rows = np.append(nonzero_rows, signal_index)
    
    values = np.ones(nonzero_rows.shape)

    #adjacency matrix as a Scipy sparse COO matrix
    a_matrix = coo_matrix((values, (nonzero_rows, nonzero_col)), 
                          [labels.size, labels.size])
    print("la shape della matrice: ", a_matrix.get_shape())
    return a_matrix
