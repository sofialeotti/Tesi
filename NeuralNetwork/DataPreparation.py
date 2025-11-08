import uproot
import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt
import tensorflow            as tf
import os

from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from spektral.data.dataset import Dataset, Graph
from spektral.utils import add_self_loops, normalized_adjacency



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
        self.df_signal     = treeS.arrays(library = "pd")
        print(self.df_signal)
        
        print("--------------------------------------------------------------------------------df_background--------------------")
        self.df_background = treeB.arrays(library = "pd", entry_stop=1000000)
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
        print("shape di X dopo concatenation: ", X.shape)

        print("--------------------------------------------------------------------------------X--------------------")
        print(X)

        print("------------------------------------------------------------Add target--------------------")  
        y = np.concatenate([np.ones(len(X_signal_normalized)),
                                np.zeros(len(X_background_normalized[:943645]))])
        
        
        
        print("--------------------------------------------------------------------------------y--------------------")
        print(y)

        print("------------------------------------------------------------Split data--------------------")
        # Split data into training and test sets
        if model_type=="GNN":
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                                    y,
                                                                                    test_size    = 0.3,
                                                                                    random_state = 42,
                                                                                    shuffle      = True)
            #splitting del dataset in tre parti uguali (train, validation, test)
        else:
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
        print("---------------------------------------------------------------------------GRAPH_PREPARATION--------------------")
        if model_type == "GNN":
            #Create the graph containing the data for the GNN training
            #splitting
            #node_features_train = self.X_train.to_numpy()
            node_features_train_pd, node_features_val_pd, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                                                self.y_train, 
                                                                                                test_size=0.5,
                                                                                                random_state=42,
                                                                                                shuffle=True)
            
            #remove the last rows of the dataframe if they contain a different number of elements
            if len(node_features_train_pd) != len(self.X_test):
                difference = len(node_features_train_pd) - len(self.X_test)
                print("lenght difference (train set-test set): ", difference)
                if difference > 0:                    
                    node_features_train_pd = node_features_train_pd.head(len(self.X_test))
                    self.y_train = self.y_train[:-difference]
                else:
                    self.X_test = self.X_test.head(len(node_features_train_pd))
                    self.y_test = self.y_test[:difference]
            
            if len(node_features_train_pd) != len(node_features_val_pd):
                difference = len(node_features_train_pd) - len(node_features_val_pd)
                print("lenght difference (train set- validation set): ", difference)
                if difference > 0:
                    node_features_train_pd = node_features_train_pd.head(len(node_features_val_pd))
                    self.y_train = self.y_train[:-difference]
                else:
                    node_features_val_pd = node_features_val_pd.head(len(node_features_train_pd))
                    self.y_val = self.y_val[:difference]
                """
                print("lunghezza di train ", len(node_features_train_pd))
                print("lunghezza di val: ", len(node_features_val_pd))
                print("lunghezza di test ", len(self.X_test))
                print("lunghezza di train labels ", len(self.y_train))
                print("lunghezza di val labels: ", len(self.y_val))
                print("lunghezza di test labels", len(self.y_test))
                """

            print("train dataframe:\n", node_features_train_pd)
            print("validation dataframe: \n", node_features_val_pd)
            print("test dataframe: \n", self.X_test)

            #transform pandas dataframes into numpy arrays
            node_features_train = node_features_train_pd.to_numpy()
            node_features_val = node_features_val_pd.to_numpy()
            node_features_test = self.X_test.to_numpy()

            print("shape di X_train:", node_features_train.shape,
              "shape di y_train: ", self.y_train.shape,
              "shape di X_val:",node_features_val.shape,
              "shape di y_val: ", self.y_val.shape,
              "shape di X_test: ", node_features_test.shape,
              "shape di y_test: ", self.y_test.shape)
            
            #create the dataset for training, validation, testing
            dataset_train = GNNDataset(node_features=node_features_train,
                                       labels= self.y_train)
            dataset_val = GNNDataset(node_features=node_features_val,
                                     labels= self.y_val)
            dataset_test = GNNDataset(node_features=node_features_test,
                                      labels= self.y_test)

            self.dataset = [dataset_train, dataset_val, dataset_test]

            """
            print("----------------------------------------------------------------------Graph_rep--------------------")
            #visual representation of the trainig graph: each node is revelation, signal and background are of different colours
            #plt.figure(figsize=(10, 10))
            #graph_rep = graph.sample(1500)
            #rel_type = ([self.df_signal, self.df_background])
            #nx.draw_spring(graph, node_size=15, node_color=rel_type)
            G = nx.Graph()
            G.add_nodes_from(node_features_train.tolist())
            #G.add_edges_from(a_train.t().tolist())
            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(6,6))
            nx.draw_networkx(G, pos, with_labels=True, node_color = 'lightblue', node_size = 800)
            plt.title("Graph of the data")
            plt.axis('off')
            if not os.path.exists("graph_rep"):
                os.makedirs("graph_rep")
            plt.savefig("graph_rep/dataset_sample.svg")
            """
            
class GNNDataset(Dataset):
    def __init__(self, 
                 node_features,
                 labels):
        self.node_features = node_features
        self.labels = tf.convert_to_tensor(labels)
        a_matrix = coo_matrix((labels.size, labels.size))
        self.a_matrix = add_self_loops(a_matrix) #aggiungo un collegamento ad ogni nodo con se stesso (matrice diagonale)
        self.a_matrix = normalized_adjacency(self.a_matrix) #normalizzazione della matrice
        super().__init__()
    def read(self):
        graph = Graph(x=self.node_features, 
                      a= self.a_matrix, 
                      y=self.labels)
        return [graph]
