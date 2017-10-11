from __future__ import print_function
import numpy as np

__author__ == 'amelie'

#TO do: make python3 compatible

class K_means(object):

    def __init__(self, data_points, clusters, version="democratic", max_iter=3):
        """
        requires np.array containing all the data_points (n , d) and
        clusters (n, d)
        """
        self.dpoints = data_points
        self.version = version
        self.n_datapoints = len(self.dpoints)
        self.clusters = clusters
        self.n_clusters = len(self.clusters)
        self.dimensions = data_points.shape[1]
        self.old_cluster = np.zeros_like(self.clusters)
        # cluster-1 for each datapoint
        self.point_cluster = np.zeros((self.n_datapoints))
        self.max_iter = max_iter
        assert (isinstance(self.max_iter, int)
                ), " number of iterations is not an integer"
        self.prob_c_given_d = np.zeros((self.n_datapoints, self.n_clusters))

    def dist(self, data_point, cluster):
        """computes eucledian distance between 2 points"""
        return np.linalg.norm(data_point - cluster)

    def sum_dist(self, data_point):
        """
        sums the distance of one data point to all the clusters
        """
        s = 0
        # import pdb; pdb.set_trace()
        for j in range(self.n_clusters):
            s += np.square(self.dist(data_point, self.clusters[j]))
        return s

    def conditional_prob(self):
        """
        computes the conditional probability that data_point belongs to cluster_n.
        returns np.array containing each conditional probability for each
        datapoimt (n_datapoints, n_clusters)
        """
        for d_point in range(self.n_datapoints):
            for j in range(self.n_clusters):
                self.prob_c_given_d[d_point][j] = (1 -
                                                   (np.square(self.dist(self.dpoints[d_point], self.clusters[j])) /
                                                    self.sum_dist(self.dpoints[d_point]))) / (self.n_clusters - 1)

        return self.prob_c_given_d

    def distances(self):
        """
        computes the distacne between each datapoint and cluster
        """
        distances = np.zeros((self.n_datapoints, self.n_clusters))
        for d in range(self.n_datapoints):
            for c in range(self.n_clusters):
                distances[d, c] = self.dist(self.dpoints[d], self.clusters[c])

        return distances

    def point_to_cluster(self):
        """
        assigns each point to a cluster according to the minimum distance if
        version == distnce otherwise the highest probability if version is
        democratic returns an np_array with the cluster at the corresponding
        data_points index ie. (2,1,1) -> d1 corresponds to cluster 2, d2
        corresponds to cluster 1, d3 corresponds to cluster1
        """
        if self.version == "democratic":
            self.point_cluster = np.argmax(self.conditional_prob(), axis=1)
        elif self.version == "eucledian":
            self.point_cluster = np.argmin(self.distances(), axis=1)
        else:
            raise ValueError("not valid function")

    def update_centroid(self):
        """
        updates centroid coordinates according to the mean of all the points
        assignt to it
        """
        self.old_cluster = self.clusters

        if self.version == "eucledian":
            for c in range(self.n_clusters):
                indexes = np.where(self.point_cluster == c)
                self.clusters[c] = np.mean(
                    np.take(self.dpoints, indexes[0], axis=0), axis=0)

        elif self.version == "democratic":
            for c in range(self.n_clusters):
                indexes = np.where(self.point_cluster == c)
                self.clusters[c] = np.sum((self.dpoints.T *
                                           self.prob_c_given_d[:, c]).T, axis=0) / np.sum(self.prob_c_given_d[:, c])

        else:
            raise ValueError('is not a valid function')

    def run(self):
        """
        runs kmeans and assigns and writes output iterations to txt file
        """
        while not (self.clusters == self.old_cluster).all or i != self.max_iter:
            self.point_to_cluster()
            self.update_centroid()
