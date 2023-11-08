import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from numpy import linalg as LA
from itertools import compress
import matplotlib.pyplot as plt
import os

def remove_assessors(csv_ratings, AssessorsToBeRemoved, write):
    """
    Function for discarding the ratings of the assessors not in agreement with the majority of the listening panel and writing a new csv that contains only the ratings in agreement.

    Args:
    csv_ratings (csv file): A csv file that contains all the temporal intensity ratings obtained during a listening test procedure
    (see Demo_Temporal_Assessment.csv for the correct format of the csv file).
    AssessorsToBeRemoved (dictionary): The dictionary containing the assessors to be removed for each stimulus (output of the function agreement_assessment). 
    write (boolean): Write the updated csv file of the time intensity ratings.
    
    Returns:
    dataframe (pandas DataFrame): The updated csv that contains all the temporal intensity ratings, except for those of the assessors in AssessorsToBeRemoved, in DataFrame form.
    """ 
    # Read the csv file as a pandas DataFrame     
    dataframe=pd.read_csv(csv_ratings)
    
    for i in AssessorsToBeRemoved:
        assessorToBeRemoved=AssessorsToBeRemoved[i]
        for j in range(len(assessorToBeRemoved)):
            dataframe.drop(dataframe[(dataframe['rating_stimulus'] ==i) & (dataframe['name'] == assessorToBeRemoved[j])].index, inplace=True)
    if write:
        # Get the parent directory's path
        parent_dir_path = os.path.dirname(os.path.realpath(__file__))
        # Select the folder where the figures will be saved
        ratings_folder_path = os.path.join(parent_dir_path, "Temporal Envelopment Data")        
        dataframe.to_csv(os.path.join(ratings_folder_path, "Updated_TemporalRatings.csv"),index=False,header = True)
        
    return dataframe


def agreement_evaluation(csv_ratings, show):
    """
    Function for assessing the agreement of a listening panel in temporal perceptual evaluation tasks.

    Args:
    csv_ratings (csv file): A csv file that contains all the temporal intensity ratings obtained during a listening test procedure
    (see Demo_Temporal_Assessment.csv for the correct format of the csv file).
    show (boolean): Visualization of the modified eigenvectors and time intensity ratings.

    Returns:
    AssessorToHold (dictionary): The assessors that are in aggreement with the majority of the listening panel.
    AssessorToRemove (dictionary): The assessors that are NOT in aggreement with the majority of the listening panel
    """ 
    # Read the csv file as a pandas DataFrame     
    dataframe=pd.read_csv(csv_ratings)
    
    # Find all the assessors
    assessors=dataframe.name.unique().tolist()

    # Find tracks
    tracks=dataframe.rating_stimulus.unique().tolist()

    # Initialize the assessors that will be removed 
    AssessorToRemove={}
    # Initialize the assessors that will be hold 
    AssessorToHold={}

    # Iterate Tracks
    for i in tracks:

        ratings_assessor=dataframe[dataframe['name']==assessors[0]]
        RatingsPerTrack=np.zeros((len(ratings_assessor[ratings_assessor['rating_stimulus']==i]),len(assessors)))
        index=0
        
        #  Iterate Assessors
        for j in assessors:
            assessorOr=dataframe[dataframe['name']==j]
            trackOr=assessorOr[assessorOr['rating_stimulus']==i]
            assessor_ratings=np.array(trackOr['rating_score'].tolist())
            RatingsPerTrack[:,index]=assessor_ratings
            index+=1

        #  Compute mean values of each assessor for each track     
        mean_vals=np.mean(RatingsPerTrack,axis=0)
        #  Normalize means to the range [-1,1]
        mean_vals=2*mean_vals/100-1
        
    #    Perform non-centered PCA to the assessors ratings
        _, ncPCA_nonCenter=pca_non_centered(RatingsPerTrack,2)   
        # Edit non-centered PCA
        ncPCA_nonCenter=np.real(np.transpose(np.array(ncPCA_nonCenter)))
    #   Initialize the features for which the clustering will be performed
        modified_eigvecs=np.zeros((2,len(assessors)))
    #   Add the corresponding features
        modified_eigvecs[0,:]=mean_vals
        modified_eigvecs[1,:]=ncPCA_nonCenter[1,:]
        modified_eigvecs=modified_eigvecs.transpose()

    #  Fit the modified eigenvectors to an Agglomerative Clustering algorithm based on the maximum distace of the elements of each cluster
        clustering_model = AgglomerativeClustering(compute_distances=True,linkage='complete').fit(modified_eigvecs)
    #  Maximum distances of each cluster
        distances_distr=clustering_model.distances_
    # Compute the threshold value based on the mean of maximum distances, above which no more clusters are considered
        T=np.mean(distances_distr)
    # Compute the clusters
        clusters = hierarchy.linkage(modified_eigvecs, method="complete")

    # Find the clusters that are below the threshold value T    
        clusters=np.zeros((1,1))
        for distance in distances_distr:
            if distance>T:
                clusters+=1

    # Fit again the data with the  number of clusters computed            
        clustering_model = AgglomerativeClustering(n_clusters=int(clusters)+1,linkage='complete').fit(modified_eigvecs)
        labels = clustering_model.labels_

    #  Compute the Euclidean distance matrix and add them to a DataFrame 
        distances_matrix=pd.DataFrame()
        for assessor1 in range(len(assessors)):
            dist_col=[]
            for assessor2 in range(len(assessors)):
                eucl_dist=LA.norm(modified_eigvecs[assessor1]-modified_eigvecs[assessor2],2)
                dist_col.append(eucl_dist)
            dist_col.append(int(labels[assessor1]))
            distances_matrix[assessors[assessor1]]=dist_col
            
        distances_matrix["Labels"]=list(labels)+["End"]
        distances_matrix["Indices"]=assessors+["Labels"]
        distances_matrix.set_index("Indices", inplace = True)
        different_labels=distances_matrix.Labels.unique().tolist()
        different_labels.pop()

        # Initialize different clusters and find the Representative cluster
        clusters_len = [labels.tolist().count(label) for label in range(max(labels) + 1)]
        max_cluster_len_idx = clusters_len.index(max(clusters_len))
        max_cluster_len = max(clusters_len)
        max_cluster = most_frequent(labels.tolist())
        representative_cluster = max_cluster if max_cluster_len > len(assessors)/2-1 else None

    # Compute outlier factor OF for each element       
        OFS=[]
        for assessor in assessors:
            assessor_cluster=distances_matrix.loc[[assessor],["Labels"]]
            assessor_cluster=int(np.array(assessor_cluster["Labels"].to_list()))

            # Compute OF Neighbour for each assessor
            OF_neighbour=1-(clusters_len[assessor_cluster]/len(assessors))  

        #   Compute OF loc for each assessor

        #   Check if there is a representative class
            if bool(representative_cluster)==True or representative_cluster==0:
        #        Check if the assessor belogns to the representative class
                if assessor_cluster==representative_cluster:
                    OF_Loc=0
                else:
                    local_dist=np.array(distances_matrix.loc[[assessor],distances_matrix.loc['Labels'] == representative_cluster])
                    OF_Loc=np.min(local_dist)
            else:
                sum_of_min_dist=[]
                for different_label in different_labels:
                    local_dist=np.array(distances_matrix.loc[[assessor],distances_matrix.loc['Labels'] == different_label])
                    local_dist=np.min(local_dist)
                    sum_of_min_dist.append(local_dist)

                OF_Loc=np.sum(sum_of_min_dist)/(len(different_labels)-1)
                
            OF=(OF_Loc+OF_neighbour)/2
            OFS.append(OF)

        #  Find outlier threshold OT
        OT=np.mean(OFS)
        
        #  Find the assessors to be removed and those to be held
        toberemoved  =[]
        tobeheld  =[]
        for assessor in range(len(assessors)):
            if OFS[assessor]>OT:
                toberemoved.append(True)
                tobeheld.append(False)
            else:
                toberemoved.append(False)
                tobeheld.append(True)
        toberemoved=list(compress(assessors, toberemoved))
        tobeheld=list(compress(assessors, tobeheld))

        AssessorToRemove[i]=toberemoved
        AssessorToHold[i]=tobeheld

        #  Visualize results
        if show:      
            visualize(modified_eigvecs, assessors, tobeheld, toberemoved,dataframe[dataframe['rating_stimulus']==i],i)

    return AssessorToHold, AssessorToRemove


def pca_non_centered (x,n_components):    
    """
    Function for performing non-centered Principal Component Analysis.

    Args:
    x (numpy array): A two-dimensional vector containing the temporal intensity ratings of all the assessors for a single audio stimulus 
    n_components (integer): The number of principal components.
    
    Returns:
    evals (numpy array): The eigenvalues of the first n_components.
    evecs (list): The eigenvectors that correspond to the ratings of each assessor.
    """ 
    matrix=np.matmul(np.transpose(x),x)
    evals , evecs = LA.eig(matrix)
    idx = np.argsort(evals)[::-1]
    idxs=idx[0:n_components]
    # Optional for computing the explained variance of the n_components    
    # var_explained=(np.sum(evals[idxs])/np.sum(evals))
    evecs = evecs[:,idxs]
    transformed_data = np.dot(x, evecs)

    return evals[idxs],evecs.tolist()

def most_frequent(List):
    """
    Function for computing the element that appears the most times in a list.

    Args:
    List (list): The list for which the computation is performed. 
    
    Returns:
    num (item): The most frequently appearing item in the list.
    """ 
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
    return num


def visualize(x, assessors, assessors_in_agreement, assessors_in_disagreement, envelopment_track,i):
    """
    Function for visualizing the non-centered eigenvectors of the temporal ratings of a single track
    and the respective temporal ratings.

    Args:
    x (numpy array): The modified eigenvectors of the non-centered PCA. 
    assessors (list): The assessors that provided temporal ratings. 
    assessors_in_agreement (tuple): The assessors that are in aggreement with the majority of the listening panel for a single track.
    assessors_in_disagreement (tuple): The assessors that are NOT in aggreement with the majority of the listening panel for a single track.
    envelopment_track (DataFrame): The ratings of all the assessors of a single track.
    """ 
    # Add eigenvectors to Dataframe
    ncPCA = pd.DataFrame(x.transpose(),columns = assessors)

    # Filter Dataframe containing eigenvectors of all assessors based on the outlier detection algorithm
    ncPCA_Agree=ncPCA.filter(items=assessors_in_agreement)
    ncPCA_Disagree=ncPCA.filter(items=assessors_in_disagreement)

    # Plot modified eigenvectors of removed and hold assessors
    
    fig, ax=plt.subplots(1, 2)
    fig.set_size_inches(14,7)
    ax[0].set_xlabel('Modified PC1',fontsize=20)
    ax[0].set_ylabel('PC2',fontsize=20)
    ax[0].set_xlim([-1.05,1.05])
    ax[0].set_ylim([-1.05,1.05])
    ax[0].tick_params(axis='both',which='major',labelsize=20)
    ax[0].set_xticks( np.arange(-1,1.05,0.5), minor=False)
    ax[0].set_yticks( np.arange(-1,1.05,0.5), minor=False)

    scatter1 =ax[0].scatter(ncPCA_Agree.loc[[0]],ncPCA_Agree.loc[[1]],s=100,color='cornflowerblue', label='Assessors in Agreement')
    scatter1 =ax[0].scatter(ncPCA_Disagree.loc[[0]],ncPCA_Disagree.loc[[1]],s=100,color='red', label='Assessors to be Removed')
    ax[0].grid(alpha=0.5)
    ax[0].axhline(0, color='black')
    ax[0].axvline(0, color='black')

    # Add a legend to the plot and adjust the distance between labels
    legend = ax[0].legend(prop={'size': 10})
    
    ax[1].set_xlabel('Timestamp',fontsize=20)
    ax[1].set_ylabel('Envelopment',fontsize=20)
    ax[1].tick_params(axis='both',which='major',labelsize=20)
    
    # Plot time intensity ratings of each hold assessor
    for assessor in assessors_in_agreement:
        envelopment_track_ass=envelopment_track[envelopment_track['name']==assessor]
        envelopment_track_ass=envelopment_track_ass['rating_score']
        timestamps=np.arange(1, len(envelopment_track_ass)+1)
        ax[1].plot(timestamps, envelopment_track_ass,color='cornflowerblue')
        ax[1].set_ylim([0,100])

    # Plot TIE Ratings of each removed assessor
    for assessor in assessors_in_disagreement:
        envelopment_track_ass=envelopment_track[envelopment_track['name']==assessor]
        envelopment_track_ass=envelopment_track_ass['rating_score']
        ax[1].plot(timestamps,envelopment_track_ass.tolist(),color='red')
        ax[1].set_ylim([0,100])

    # Get the parent directory's path
    parent_dir_path = os.path.dirname(os.path.realpath(__file__))
    # Select the folder where the figures will be saved
    figures_folder_path = os.path.join(parent_dir_path, "Temporal Envelopment Data")
    # Save figures
    plt.savefig(os.path.join(figures_folder_path,'plot_{}.png'.format(i)))
    plt.show()