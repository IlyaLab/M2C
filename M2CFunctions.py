from numpy import *
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hcluster
from scipy.integrate import quad
from scipy.cluster.hierarchy import to_tree


def MyGaussian(x, u=1.0, d=1.0, normal=True):
    if normal:
        return (1.0/(d*sqrt(2*pi)))*exp(-1.0*(x-u)**2.0/(2.0*d**2.0))
    else:
        return exp(-1.0*(x-u)**2.0/(2.0*d**2.0))

#Dynamic Mixture Model Object which encapsulates at multivariate mixture model of gaussians with a uniform noise term
#This object can automatically fit itself to data in a number of ways.
class DMM(object):
    #Data: Raw data underlying the mixture model (1 dimensional)
    #Density: a Kernel density estimate of the data (option)
    #interval: interval on which the data is defined. Otherwise the interval is defined as [min(data)-1, max(data)+1]
    #noise_bais: a probability threshold between 0 and 1 for a data point to be assigned to a gaussian cluster instead of to the noise cluster
    #            for example, if noise_bias is .5, the probability of a cluster emitting a data point must be greater than 50% for that point
    #            to be assigned to the cluster.
    def __init__(self, data, density = None, interval = None, noise_bias = None):
        self.data = data
        self.N = len(data)
        if interval == None:
            self.min_data = min(data)-1 #to avoid edge effects
            self.max_data = max(data)+1 #to avoid edge effects
            self.interval = (self.min_data, self.max_data)
        else:
            self.min_data = interval[0]
            self.max_data = interval[1]
            self.interval = interval
        self.empty_clusters = []
        self.ClusterListList = []
        self.density = density
        self.ClusterDict = {}
        self.q = None
        self.means = None
        self.weights = None
        self.stdevs = None
        self.noise_bias = noise_bias
        for i in range(self.N):
            self.ClusterDict[data[i]] = -1
    
    #Uniform distribution
    def uniform(self):
        return 1.0/(self.interval[1]-self.interval[0]-2)

    #Guassian clusters are seeded at the maxima of a kernel density estimate with the standard deviation determined by the space between the two adjacent minima.
    #Uniform_weight_factor: initial estiamte of the uniform noise term
    #h: default bandwidth for the KDE
    #dsteps = see Kernel Density Estimate function
    #if extra_info == true, function returns density and local maxima as well as initial cluster estimate.
    def seed_initial_clusters(self, uniform_weight_factor = .25, h=4.0, dsteps = 'auto', extra_info = False):
        
        self.density = self.KernelDensityEstimate(data=self.data, h=h, dsteps = dsteps, interval = self.interval)[0]
            
        if dsteps == 'auto':
            dsteps = len(self.density)
            
        interval = self.interval
        step_size = (interval[1]-interval[0])/(1.0*dsteps)
        density = self.density
        local_maxima = []
        local_minima = []
        peak_min = .0001
        start_flat = 0
        for i in range(1, len(density)-1, 1):
            #Maxima
            if density[i]>density[i+1] and density[i] > density[i-1] and density[i] > peak_min:
                local_maxima.append((i*step_size, density[i]))
            #maxima flat to the left
            elif density[i]>density[i+1] and density[i] == density[i-1] and density[i] > peak_min:
                end_flat = i
                ind = int(start_flat+(end_flat-start_flat)/2.0)
                local_maxima.append((ind*step_size, density[ind]))
            #Maxima flat to the right
            elif density[i]==density[i+1] and density[i] > density[i-1] and density[i] > peak_min:
                start_flat = i
            #Minima
            elif density[i]<density[i+1] and density[i] < density[i-1]:
                local_minima.append((i*step_size, density[i]))
            #Minima: Flat to the left
            elif density[i]<density[i+1] and density[i] == density[i-1]:
                end_flat = i
                ind = int(start_flat+(end_flat-start_flat)/2.0)
                local_minima.append((ind*step_size, density[ind]))
            #Minima: Flat to the right
            elif density[i] == density[i+1] and density[i] < density[i-1]:
                start_flat = i        

        if len(local_minima) == 0:
            local_minima.append((0, density[0]))
        
        #case where no local max found: use the maxima of the density estimate
        if len(local_maxima) == 0:
            local_maxima.append((density.index(max(density)), max(density)))
           
        mx = local_maxima[0]
        mn = local_minima[0]
        #max before local min at start of density
        if mx[0] < mn[0]:
            local_minima.insert(0, (0, density[0]))

        mx = local_maxima[-1]
        mn = local_minima[-1]
        #no min after max at end of density
        
        if mx[0] > mn[0]:
            local_minima.append((len(density), density[-1]))

        #set means to be local maxima and k
        self.k = len(local_maxima)
        self.means = array([1.0*m[0] for m in local_maxima])
        #estimate stdevs from distance between local max and min
        self.stdevs = []
        for i in range(len(local_maxima)):
            self.stdevs.append((local_minima[i+1][0] - local_minima[i][0])/2.0)
        
        self.weights = [m[1] for m in local_maxima]
        self.weights = [(1.0-uniform_weight_factor)*w/sum(self.weights) for w in self.weights]
        self.weights.append((1.0*uniform_weight_factor))
        self.weights = array(self.weights)
        if not extra_info:
            return (self.weights, self.means, self.stdevs)
        else:
            return self.weights, self.means, self.stdevs, density, local_maxima

    #Expectation Step of the EM optimization algorithm
    #returns emission matrix, q
    def Expectation(self, weights = None, means = None, stdevs = None):
        N = self.N
        if weights == None:
            weights = self.weights
        k = len(weights)
        
        if means == None:
            means = self.means
            
        if stdevs == None:
            stdevs = self.stdevs
            
        q = zeros((N, k))

        for n in range(N):
            d = self.data[n]
            for j in range(k):
                Z = weights[k-1]*self.uniform()+sum([weights[i]*MyGaussian(d, u=means[i], d = stdevs[i]) for i in range(k-1)])
                if j == k-1:
                    q[n, j] = weights[j]*self.uniform()/Z
                else:
                    q[n, j] = weights[j]*MyGaussian(d, u=means[j], d = stdevs[j])/Z
        return q

    #Calcualte the weights of each mixture model component by summing over the emission matrix
    def CalculateWeights(self, q = None):
        if q == None:
            q = self.q
            
        k = len(q[0, :])
        N = self.N

        weights = zeros((k, 1))
        for j in range(k):
            weights[j] = 1.0/(N)*sum(q[:, j])

        return weights
    
    #Maximization step of the EM algorithm - returns optimized gaussians and noise term based upon an emission matrix q
    def Maximization(self, q = None):
        if q == None:
            q = self.q
            
        data = self.data
        N = self.N
        k = len(q[0, :])
        means = zeros((k-1, 1))
        stdevs = zeros((k-1, 1))
        for j in range(k-1):
            E_sum = sum([q[n, j] for n in range(N)])
            if E_sum != 0:
                means[j] = sum(data[n]*q[n, j] for n in range(N))/E_sum
                stdevs[j] = sqrt(sum([q[n, j]*(data[n]-means[j])**2 for n in range(N)])/E_sum)

            if isnan(stdevs[j]) or stdevs[j] < sqrt(1/(2*3.141592654)) or not isfinite(stdevs[j]):
                stdevs[j] = sqrt(1/(2*3.141592654))

        return (means, stdevs)

    #Assigns data points to a given cluster based upon theargmax of the emission matrix.
    def AssignClusters(self, q = None):
        ClusterDict = {}
        if q == None:
            q = self.q
        if self.noise_bias == None:
            for n in range(self.N):
                d = self.data[n]
                cluster = argmax(q[n, :])
                ClusterDict[d] = cluster
        else:
            k = len(q[0, :])
            for n in range(self.N):
                d = self.data[n]
                maxcluster = argmax(q[n, :])
                q_cluster = q[n, maxcluster]
                q_noise = q[n, -1]
                if maxcluster != k-1 and q_noise/q_cluster < self.noise_bias:
                    ClusterDict[d]=maxcluster
                else:
                    ClusterDict[d]=k-1
        return ClusterDict
   
    #Encapsulates one iteration of the EM algorithm
    def EMIteration(self, weights=None, means = None, stdevs = None):
     
        q = self.Expectation(weights = weights, means = means, stdevs = stdevs)
        weights = self.CalculateWeights(q)
        means, stdevs = self.Maximization(q)
        CD = self.AssignClusters(q)
        if CD == None:
            raise RuntimeError("CD == None in EMIteration")

        return (weights, means, stdevs, CD)

    #EM Algorithm until clusters stop changing
    def EMLoop(self, weights=None, means = None, stdevs = None, extra_iterations = 0):
        iterate = True
        weights, means, stdevs, CD1 = self.EMIteration(weights, means, stdevs)
        iters =0
        while iterate and iters<extra_iterations:
            iterate = False
            
            weights, means, stdevs, CD2 = self.EMIteration(weights, means, stdevs)
            if CD1 == None:
                iterate = True
            else:
                for key in CD1:
                    if CD1[key] != CD2[key]:
                        iterate = True
            if CD2 == None:
                raise RuntimeError("CD2==None Whooops")
            
            CD1 = dict(CD2)
            if CD1 == None:
                raise RuntimeError("This makes no sense!")
            if not iterate:
                iters += 1
        
        return weights, means, stdevs, CD1
    
    #Calculates the Loglikelihood of the model producing the data
    def LogLikelihood(self, weights = None, means = None, stdevs = None):
        if weights == None:
            weights = self.weights
        if means == None:
            means = self.means
        if stdevs == None:
            stdevs = self.stdevs
            
        N = self.N
        k = len(weights)

        log_likelihood = 0
        for n in range(N):
            d = self.data[n]
            #Z = weights[-1]*self.uniform()+sum([weights[j]*MyGaussian(d, u=means[j], d=stdevs[j]) for j in range(k-1)])
            log_likelihood+=log(weights[-1]*self.uniform()+sum([weights[j]*MyGaussian(d, u=means[j], d=stdevs[j]) for j in range(k-1)]))

        return log_likelihood
    
    #Returns a list of lists each representing a cluster and containing the assigned data points inside
    def GenerateClusterList(self, ClusterDict = None, k = None):
        if ClusterDict == None:
            ClusterDict = self.ClusterDict
        if ClusterDict == None:
            raise RuntimeError("ClusterDict == None! OH NO!")
        if k == None:
            k = len(self.weights)
        ClusterList = [[] for i in range(k)]
        for d in self.data:
            try:
                ClusterList[int(ClusterDict[d])].append(d)
            except KeyError:
                print "Key Error for", d
                print ClusterDict.keys()
                raise KeyError
        
        return ClusterList

    #Similar to the above, but will recalculate the expectation matrix
    def GetClusterList(self):
        if self.q == None:
            q = self.Expectation()
        else:
            q = self.q
        return self.GenerateClusterList(self.AssignClusters(q=q))

    #Converts a dictionary into a cluster list. For internal use.
    def GenerateClusterDictFromList(self, ClusterList = None):
        if ClusterList == None:
            ClusterList = self.ClusterList
        ClusterDict = {}
        for i in range(len(ClusterList)):
            cluster = ClusterList[i]
            for d in cluster:
                ClusterDict[d] = i

        return ClusterDict
    
    #Baysian Information Criteria of the model with respect to the underlying data
    def BIC(self, weights = None, means = None, stdevs = None):
        if weights == None:
            weights = self.weights
        if means == None:
            means = self.means
        if stdevs == None:
            stdevs = self.stdevs

        k = len(weights)
        
        return -2*self.LogLikelihood(weights, means, stdevs)+(3*(k-1)+1)*log(len(self.data))
    
    #Kernel Density estimate of the data
    #Data: raw for the KDE
    #h: KDE bandwidth
    #k_type: mixture model kernel. Supported options are 'triangle', 'parabolic', 't-distribution', 'normal', and 'uniform'
    #d_steps: number of discrete steps in the KDE. default is 10x the interval the data is defined on
    #K: a custom kernel of the form K(u) can be passed in. This overrides the k_type parameter
    #interval: interval the data is defined on
    def KernelDensityEstimate(self, data = None, h=10.0, k_type = 'normal', dsteps = 'auto', K = None, interval = None):
        if data == None:
            data = self.data
        N = len(data)
        if dsteps == 'auto' and interval == None:
            dsteps = int(10*(max(data)-min(data)))
            interval = (min(data), max(data))
        elif dsteps == 'auto':
            dsteps = int(10*(interval[1] - interval[0]))
    
        if K == None:
            if k_type == 'normal':
                def K(u):
                    return 1/sqrt(2*3.141592654)*exp(-.5*u*u)
            elif k_type == 'uniform':
                def K(u):
                    if abs(u) <=1:
                        return .5
                    else:
                        return 0.0
            elif k_type == "tri" or k_type == "triangle" or k_type == "triangular":
                def K(u):
                    if abs(u)<=1:
                        return 1-abs(u)
                    else:
                        return 0
            elif k_type == "parabolic" or k_type == "evan":
                def K(u):
                    if abs(u) <=1:
                        return -.75*(u*u-1)
                    else:
                        return 0
            elif k_type == "t" or k_type == "t-distribution":
                def K(u):
                    return 1.0/1.77245*(1+u*u)**(-1)
            else:
                raise TypeError("Invalid Kernal Type")

        step = (interval[1]-interval[0])/(1.0*dsteps)
        x_range = [interval[0]+i*step for i in range(int(dsteps))]
        density = [1.0/(N*h)*sum([K((x-xi)/h) for xi in data]) for x in x_range]
        
        return density, x_range
    #Akaike Information Criteria for the model on the data
    def AIC(self, weights = None, means = None, stdevs = None):
        if weights == None:
            weights = self.weights
        if means == None:
            means = self.means
        if stdevs == None:
            stdevs = self.stdevs
        return 2*(3*(self.k-1)+1)-2*self.LogLikelihood(weights = weights, stdevs = stdevs, means = means)
    #Akaike Information Criteria for the model on the data
    def AICc(self, weights = None, means = None, stdevs = None):
        if weights == None:
            weights = self.weights
        if means == None:
            means = self.means
        if stdevs == None:
            stdevs = self.stdevs
        n = len(self.data)
        p = 3*(self.k-1)+1
        if (n-p-1) == 0:
            return 9999999999999 #To avoide infinities
        else:
            return self.AIC(weights, means, stdevs)+2.0*p*(p+1)/(n-p-1)

        return weights, means, stdevs, CD
    
    #Automatically Runs the dynamic mixture model on the data
    def AutoRun(self, h, uniform_weight_factor = .25, extra_iterations = 0, extra_info = False, print_outs = False):

        if not extra_info:
            weights, means, stdevs, = self.seed_initial_clusters(h=h, extra_info = extra_info)
        else:
            weights, means, stdevs, density, local_max = self.seed_initial_clusters(h=h, extra_info = extra_info)
        
        iterate = True
        extra_i = 0
        auto_run_iterations = 0
        if print_outs:
            print "Initial State: weigths, means, stdevs", (weights, means, stdevs)
            #print "len: weigths, means, stdevs", (len(weights), len(means), len(stdevs))
        while(iterate):
            auto_run_iterations += 1
            iterate = False
            (weights, means, stdevs, CD) = self.EMIteration()
            
            for key in CD:
                if CD[key] != self.ClusterDict[key]:
                    iterate = True
                    self.ClusterDict[key] = CD[key]
                    extra_i = 0
                    
            k = len(weights)
            ClusterList = [[] for i in range(k)]
            for d in self.data:
                ClusterList[self.ClusterDict[d]].append(d)

            to_remove = []
            for i in range(k-1):
                cluster = ClusterList[i]
                if len(cluster) < 1:
                    weights[i] = 0
                    to_remove.append(i)

            self.weights = array([weights[i] for i in range(k) if i not in to_remove])
            self.weights = self.weights/sum(self.weights)
            self.means = array([means[i] for i in range(k-1) if i not in to_remove])
            self.stdevs = array([stdevs[i] for i in range(k-1) if i not in to_remove])
            q = self.Expectation()
            self.ClusterDict = self.AssignClusters(q)
            self.ClusterList = self.GenerateClusterList()
            self.k = self.k - len(to_remove)

            if iterate == False and extra_i <= extra_iterations:
                iterate = True
                extra_i+=1
        if print_outs:
            print "Autorun completed after "+str(auto_run_iterations)+" EM iteterations"
        
        if not extra_info:
            return self.weights, self.means, self.stdevs, self.ClusterDict
        else:
            return self.weights, self.means, self.stdevs, self.ClusterDict, density, local_max


#distance metric = INT[|f(x)-g(x)|]dx from I[0] to I[1]
#p1 = (w1, m1, d1), etc
def Cdist(p1, p2, I, dx = .01):
    w1, m1, d1, noise= p1
    w2, m2, d2, noise  = p2
    #INT = sum([abs(w1*MyGaussian(x, m1, d1)-w2*MyGaussian(x, m2, d2))*dx for x in arange(I[0], I[1], dx)])
    def dfunc(x):
        return abs(w1*MyGaussian(x, m1, d1)-w2*MyGaussian(x, m2, d2))
    INT = quad(dfunc, I[0], I[1])
    return INT[0]

#Given a list of leaves in the cluster tree, CreateOverlapDict creates a dictionary which takes the
#(start, end) of a cluster and returns all clusters which overlap with it
def CreateOverlapDict(leaves, CD):
        #check for overlap in leaves -> store in dictionary of lists
        overlap_dict = {}
        N = len(leaves)
        for i in range(N):
            for j in range(i+1, N):
                if leaves[i].is_leaf() and leaves[j].is_leaf():
                    id1 = leaves[i].id
                    c1, p1 = CD[id1]
                    c1_min = min(c1)
                    c1_max = max(c1)
                    id2 =leaves[j].id
                    c2, p2 = CD[id2]
                    c2_min = min(c2)
                    c2_max = max(c2)
                    if (c1_min <= c2_max and c1_max >= c2_max) or (c2_min <= c1_max and c2_max >= c1_max):
                        if (c1_min,c1_max) in overlap_dict:                    
                            overlap_dict[(c1_min,c1_max)].append((c2_min,c2_max))
                        else:
                            overlap_dict[(c1_min,c1_max)]=[(c2_min,c2_max)]
                        if (c2_min,c2_max) in overlap_dict:
                            overlap_dict[(c2_min,c2_max)].append((c1_min,c1_max))
                        else:
                            overlap_dict[(c2_min,c2_max)]=[(c1_min,c1_max)]
        return overlap_dict
        
#Mapping from the (start, end) of a cluster to the tree element
def CreateLeafDict(leaves, CD):
    leaf_dict = {}
    for leaf in leaves:
        if leaf.is_leaf():
            id1 = leaf.id
            c, p = CD[id1]
            if len(c)>0:
                c_min = min(c)
                c_max = max(c)
                leaf_dict[(c_min, c_max)] = leaf    
    return leaf_dict




def CreateOverlapTree2(overlap_dict, overlap_tree_dict = {}, extra_info = False):
    
    #sort clusters by size
    cluster_list = [(key[1]-key[0], key) for key in overlap_dict]
    cluster_list.sort()
    cluster_list.reverse()
    cluster_list = [item[1] for item in cluster_list]
    if extra_info:
        print "Createing Overlap Tree for:", cluster_list
    #create a list of all non-overlapping sets of clusters from the cluster_list
    #Naive implementation which is not fully optimized, but good enough
    overlap_tree = []
    for base_cluster in cluster_list:
        #list clusters which do not overlap with the base cluster
    
        non_exclusive_options = [c for c in cluster_list if c not in overlap_dict[base_cluster] and c!=base_cluster]
        non_exclusive_options.sort()
        if extra_info:
            print "Non exclusive options", non_exclusive_options
        #Case 1: Everything in the cluster list overlaps with the base_cluster
        #This is the base case and included the one-cluster case
        if len(non_exclusive_options) == 0:
            if extra_info:
                print "base case: adding "+str(base_cluster)+" to overlap tree"
            overlap_tree.append([base_cluster])
            
        #Case 2: 
        else:
            if extra_info:
                print "recursive call on overlap tree"
            #Calculate the sub-overlap dict for all non-overlapping options.
            tuple_list = tuple(non_exclusive_options)
            #print "tuple_list", tuple_list
            #print "overlap_tree_dict", overlap_tree_dict
            if tuple_list in overlap_tree_dict:
                sub_overlap_list = overlap_tree_dict[tuple_list]
            else:
                sub_overlap_dict = {c1:[c2 for c2 in overlap_dict[c1] if c2 in non_exclusive_options] for c1 in non_exclusive_options}
                #recursive call to CreateOverlapTree
                sub_overlap_list = CreateOverlapTree2(sub_overlap_dict, overlap_tree_dict = overlap_tree_dict, extra_info = extra_info)
                overlap_tree_dict[tuple_list] = sub_overlap_list
            for sublist in sub_overlap_list:
                overlap_tree.append([base_cluster]+sublist)
    
    #Overlap tree will contain duplicates - remove them
    if extra_info:
        print "overlap tree before pruning", overlap_tree
    for subtree in overlap_tree:
        subtree.sort()
    for subtree in overlap_tree:
        while overlap_tree.count(subtree)>1:
            overlap_tree.remove(subtree)
    if extra_info:
        print "overlap_tree after pruning", overlap_tree
    return overlap_tree
        
        
    
    
#Places clusters into a tree represented as a list of lists.
#Each level of the tree represents a set of non-overlapping clusters.
#All such lists of non-overlapping clusters are enumerated here.
def CreateOverlapTree(overlap_dict, CD, extra_info = False):
    overlap_list = [key for key in overlap_dict]
    overlap_list.sort() #sort clusters from left to right - do not re-order or clusters may be skipped.

    if extra_info:
        print "overlapping leaves"
        print overlap_list
    
    overlap_tree = []
    while(len(overlap_list) > 0):
        exclusive_clusters = [overlap_list[0]]+overlap_dict[overlap_list[0]]
        new_overlap_tree = [] #will later repalce overlap_tree
        if extra_info:
            print "Exclusive clusters:", exclusive_clusters
            if len(overlap_tree)==0:
                print "base case"
        for c in exclusive_clusters:
            if extra_info:
                print "evaluating cluster", c
            #initial case
            if len(overlap_tree)==0:
                new_overlap_tree.append([c])
            else:
                for sub_list in overlap_tree:
                    
                    #if c not exclusive with anything in the sublist, add sublist+[c] to new_overlap_tree
                    overlap_testing = [(c in overlap_dict[s] or c == s) for s in sub_list] 
                    if extra_info:
                        print "overlap_testing", overlap_testing
                        print "is "+str(c) +"in ", [overlap_dict[s] for s in sub_list]
                    if True not in overlap_testing and c not in sub_list:
                        new_overlap_tree.append(sub_list+[c])
                        if extra_info:
                            print "adding "+str(c)+ " to", sub_list
            if c in overlap_list:
                overlap_list.remove(c)
        overlap_tree = list(new_overlap_tree)
        
    return overlap_tree

def CalculateIC(leaves, data, CD):
    noises = 0
    weights = 0
    for l in leaves:
        #print "l", l
        #print "leaf_dict.keys()", leaf_dict.keys()
        #print "leaf_dict[l]", leaf_dict[l]
        #print "l.id", l.id
        #print "CD.keys()", CD.keys()
        c ,p = CD[l.id]
        w, m, d, noise = p
        weights += w
        noises += noise
    noise_weight = noises/len(leaves)
    if (weights+noise_weight) > 1:
        weight_factor = 1./(weights+noise_weight)
    else:
        weight_factor = 1.
    likelihood_by_data_points = []
    for x in data:
        L = 0.0+weight_factor*noise_weight
        for key in leaves:
            c ,p = CD[l.id]
            w, m, d, noise = p
            L+=1.0*weight_factor*w*MyGaussian(x, m, d)

        likelihood_by_data_points.append(L)
    LL=sum(log(likelihood_by_data_points))
    params = 3*len(leaves)+1
    AIC  = 2*params - 2*LL
    n = len(data)
    if n-params-1<=0:
        AICc = (-log(0))
    else:
        AICc = AIC+2*params*(params+1)/(n-params-1)
    BIC=-2*LL+3*(len(leaves)+1)*log(len(data))
    return BIC, AIC, AICc
#Recursively flatten the tree by choosing the set of non-overlapping clusters which
#minimize the information criteria.
def FlattenTree(n, data, CD, interval, extra_info = False, print_outs = False):
    #base case: n is leaf
    if n.is_leaf():
        return ([n], None)
    else:
        #recursive call
        leaves = FlattenTree(n.get_right(), data, CD, interval, extra_info)[0]+FlattenTree(n.get_left(), data, CD, interval, extra_info)[0]
        
        leaf_dict = CreateLeafDict(leaves, CD)
        
        #check if leaves overlap
        if extra_info:
            leaf_intervals = [(min(CD[l.id][0]), max(CD[l.id][0])) for l in leaves if len(CD[l.id][0])>0]
            leaf_intervals.sort()
            print "flattenning:", leaf_intervals
        overlap_dict = CreateOverlapDict(leaves, CD)
        
        #No overlap:
        if len(overlap_dict.keys())==0:
            #calculate IC
            BIC, AIC, AICc = CalculateIC(leaves, data, CD)
            return leaves, AICc
        #multiple overlaps
        else:
            overlap_tree = CreateOverlapTree2(overlap_dict, extra_info=extra_info)
            non_overlapping_leaves = [(min(CD[l.id][0]), max(CD[l.id][0])) for l in leaves if (min(CD[l.id][0]), max(CD[l.id][0])) not in overlap_dict]

            for sublist in overlap_tree:
                sublist+=[c for c in non_overlapping_leaves if c not in sublist]
                sublist.sort()
            if extra_info:
                print "overlap tree before pruning", overlap_tree
                
            for L in overlap_tree:
                L.sort()
            for L in overlap_tree:
                if overlap_tree.count(L)>1:
                    overlap_tree.remove(L)
                    
            if extra_info:
                print "overlap tree after pruning", overlap_tree
            
            
            #calculate lower and upper bounds for cluster data points
            lower_bounds = []
            upper_bounds = []
            for sublist in overlap_tree:
                lower_bounds.append(min([c[0] for c in sublist]))
                upper_bounds.append(max([c[1] for c in sublist]))
            normalizer = 1.0
            
            #If all IC's are infinite, reduce the data set iterative by 20% until we get non-infinite values.
            #In practice this is rarely necessary
            iteration = 0
            while  iteration == 0 or (array(AICcs) == inf).all() or (array(LLs) == 0).all():
                sample_size = int(len(data)*(.8**iteration))
                #print "sample size", sample_size
                #raw_input("press enter to continue")
                random_indices = range(len(data))
                random.shuffle(random_indices)
                random_indices = random_indices[:sample_size]
                restricted_data = [data[i] for i in random_indices]
                
                LLs = []
                weights = []
                BICs = []
                AICcs = []
                noises = 0
                
                for sublist in overlap_tree:
                    likelihood_by_data_points = []
                    weights = 0
                    noise = 0
                    for key in sublist:
                        c ,p = CD[leaf_dict[key].id]
                        w, m, d, noise = p
                        weights+=w
                        #print "(c, p)", c, p
                        noises+=noise
                    #print "noises", noises
                    #raw_input("press enter to cont")
                    noise_weight = noises/len(sublist)
                    
                    if (weights+noise_weight)>normalizer:
                        weight_factor = 1.0*normalizer/(weights+noise_weight)
                    else:
                        weight_factor = 1.0
                    
                    for x in restricted_data:
                        L = 1.0*weight_factor*noise_weight
                        #print "L", L
                        for key in sublist:
                            
                            c ,p = CD[leaf_dict[key].id]
                            w, m, d, noise = p
                            L+=1.0*weight_factor*MyGaussian(x, m, d)
                            #print "key, params, prob", (key, p,weight_factor, MyGaussian(x, m, d))
                        #print "L", L
                        #print "w factor, noise weight, noises, weight", (weight_factor, noise_weight, noises, weights)
                        likelihood_by_data_points.append(L)
                        
                    LLs.append(sum(log(likelihood_by_data_points)))
                    params = 3*len(sublist)+1
                    AIC  = 2*params - 2*LLs[-1]
                    n = len(data)
                    if n-params-1<=0:
                        AICcs.append(-log(0))
                    else:
                        AICcs.append(AIC+2*params*(params+1)/(n-params-1))
                    BICs.append(-2*LLs[-1]+3*(len(sublist)+1)*log(len(data)))
                    
                    iteration += 1
                
                if extra_info:
                    print 'sublist', sublist
                    print "len restricted data", len(restricted_data)
                    print "LLs", LLs
                    print "BICs", BICs
                    print "AICcs", AICcs
                    print "weights", weights
            
            
            
            ind = AICcs.index(min(AICcs))
            new_leaves = [leaf_dict[k] for k in overlap_tree[ind]]
            if extra_info:
                print "AICc minimized (min AICc, ind)=", (min(AICcs), ind)
                
            
            if extra_info:
                print "new leaves", [(min(CD[l.id][0]), max(CD[l.id][0])) for l in new_leaves]
            return new_leaves, min(AICcs)

#Error Checking Method to Ensure that no clusters overlap
def CheckForClusterOverlap(all_leaves, leaves, CD, data, interval, extra_info = False, print_outs = False):
    full_overlap_dict = CreateOverlapDict(all_leaves, CD)
    
    leaf_intervals = [(min(CD[l.id][0]), max(CD[l.id][0])) for l in leaves]
    if print_outs:
        print "initial clusters:", leaf_intervals
    for l1 in leaf_intervals:
        if leaf_intervals.count(l1) >1:
            print "Cluster duplication:", l1
            raise RuntimeError("Final Cluster contains duplicates")
        for l2 in leaf_intervals:
            if l1 != l2 and l1 in full_overlap_dict and l2 in full_overlap_dict[l1]:
                print "overlapping intervals", (l1, l2)
                raise RuntimeError("Final Clusters contain overlaps")
                
                
                
#Given a set of clusters, this method ensures that none of them overlap
#Then, it looks at all additional clusters and ensures that any clusters which were
#omitted by the greedy tree flattening algorithm are added. This works the same way
#the tree flattenning algorithm works, namely by minimizing the information criteria for different
#non-overlapping combinations of omitted clsuters.
def TreeFixer(all_leaves, leaves, CD, data, interval, extra_info = False, print_outs = False):
    leaf_dict = CreateLeafDict(all_leaves, CD)
    full_overlap_dict = CreateOverlapDict([l for l in all_leaves if l.is_leaf()], CD)
    #leaf_overlap_dict = CreateOverlapDict(leaves, CD)
    all_intervals = [(min(CD[l.id][0]), max(CD[l.id][0])) for l in all_leaves if l.is_leaf()]
    leaf_intervals = [(min(CD[l.id][0]), max(CD[l.id][0])) for l in leaves if l.is_leaf()]
    if print_outs:
        print "checking for overlap"
    CheckForClusterOverlap([l for l in all_leaves if l.is_leaf()], leaves, CD, data, interval, extra_info, print_outs)
    
    missed_clusters = []
    for c in all_intervals:
        if c not in leaf_intervals:
            if print_outs:
                print "checking", c
            missed = True
            for l in leaf_intervals:
                if l in full_overlap_dict and c in full_overlap_dict[l]:
                    missed = False
            if missed:
                if print_outs:
                    print "missed cluster", c
                missed_clusters.append(c)
    
    if print_outs:
        print "initially found clusters before tree fixing:", len(leaf_intervals)
        print "total missed clusters:", len(missed_clusters)
    
    overlap_list = [key for key in full_overlap_dict if key in missed_clusters]
    overlap_list.sort()
    if print_outs:
        print "creating overlap dict"
    partial_overlap_dict = CreateOverlapDict([leaf_dict[l] for l in overlap_list], CD)
    if print_outs:
        print "creating overlap tree"
    overlap_tree = CreateOverlapTree2(partial_overlap_dict, extra_info=extra_info)
    if print_outs:
        print "overlap_tree", overlap_tree
    non_exclusive_clusters = [c for c in overlap_list if c not in partial_overlap_dict.keys()]
    if print_outs:
        print "non_exclusive_clusters", non_exclusive_clusters

    for sublist in overlap_tree:
        sublist+=[c for c in leaf_intervals if c not in sublist]+non_exclusive_clusters
        sublist.sort()
    
    
    if len(overlap_tree)>0:
        if print_outs:
            print "Tree Fixing Algorithm Needed"
        BICs = []
        AICcs = []
        weights = []
        LLs = []
        noises = []
        for sublist in overlap_tree:
            likelihood_by_data_points = []
            weights.append(0.0)
            noises.append(0.0)
            for l in sublist:
                
                c ,p = CD[leaf_dict[l].id]
                w, m, d, noise = p
                weights[-1]+=w
                noises[-1]+=noise
            avg_noise = mean(noises[-1])
            if weights[-1]>1:
                weight_factor = 1.0/(weights[-1]+avg_noise)
            else:
                weight_factor = 1.0
            
            for x in data:
                L = 1.0*weight_factor*avg_noise
                for l in sublist:
                    c ,p = CD[leaf_dict[l].id]
                    w, m, d, noise = p
                    L+=weight_factor*w*MyGaussian(x, m, d)
                likelihood_by_data_points.append(L)
            LLs.append(sum(log(likelihood_by_data_points)))
            params = 3*len(sublist)+1
            AIC  = 2*params - 2*LLs[-1]
            n = len(data)
            if n-params-1 <= 0:
                AICcs.append(-log(0))
            else:
                AICcs.append(AIC+2*params*(params+1)/(n-params-1))
            BICs.append(-2*LLs[-1]+3*(len(sublist)+1)*log(len(data)))
        
        ind = AICcs.index(min(AICcs))
        new_leaves = [leaf_dict[k] for k in overlap_tree[ind]]
        CheckForClusterOverlap(all_leaves, new_leaves, CD, data, interval)
        return new_leaves, AICcs[ind]
    else:
        if print_outs:
            print "Tree Fixing Algorithm Not Needed"
        CheckForClusterOverlap(all_leaves, leaves, CD, data, interval)
        return leaves, None

#Takes a list of clusters of the form [[cluster_mutations]....] and a 
#list of parameters of the form [(weight, mean, stdev, noise weight)].
#the ith cluster should correspond to the ith paramater element.
#Parameters for clusters with the same mutations inside of them are averaged together
#Clusters with fewer mutations than min_mut_count are excluded
def CreateClusterSet(cluster_list, params_list, min_mut_count = 15, print_outs = False):
    #remove any empty clusters
    while cluster_list.count([]) > 0:
        ind = cluster_list.index([])
        cluster_list.pop(ind)
        params_list.pop(ind)
        
    cluster_set = []
    params_set = []
    if print_outs:
        print "creating cluster set"
    #Averages params of clusters that contain the same points and exclude clusters with too few mutations
    for i in range(len(cluster_list)):
        c = cluster_list[i]
        if c not in cluster_set and len(c)>=min_mut_count:
            if cluster_list.count(c)>1:
                duplicate_params = [params_list[i] for i in range(len(params_list)) if cluster_list[i]==c]
                #calculate average cluster params
                w_avg = mean([p[0] for p in duplicate_params])
                m_avg = mean([p[1] for p in duplicate_params])
                s_avg = mean([p[2] for p in duplicate_params])
                n_avg = mean([p[3] for p in duplicate_params])
                cluster_set.append(c)
                params_set.append((w_avg, m_avg, s_avg, n_avg))
            elif cluster_list.count(c) == 1:
                cluster_set.append(c)
                params_set.append(params_list[i])
    
    return (cluster_set, params_set)

#Uses Scipy's Heirarchical Clustering Method to create a tree of clusters based upon the
#distance between the gaussian curves each cluster represents
def CreateHeiarchicalTree(cluster_set, params_set, interval, print_outs = False):
    L = len(cluster_set)
    if L>1:
        Dmatrix = zeros((L,L))
        leaf_index_cluster_dict = {}
        if print_outs:
            print "calculating function distances"
        for i in range(L):
            p1 = params_set[i]
            c1 = cluster_set[i]
            
            leaf_index_cluster_dict[i] = (c1, p1)
            for j in range(L):
                
                p2 = params_set[j]
                Dmatrix[i,j] = Cdist(p1, p2, interval)
        
        if print_outs:
            print "converting to square form"
        Dmatrix_c = squareform(Dmatrix)
        if print_outs:
            print "hclustering"
        linkageMatrix = hcluster.linkage(Dmatrix_c)
        if print_outs:
            print "creating tree"
        Root, node_list = to_tree(linkageMatrix, rd=True)
    else:
        raise Exception("Cannot generate a tree from a single cluster")
    return Root, node_list, leaf_index_cluster_dict