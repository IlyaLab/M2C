import MultiscaleMutationClusteringScript as M2C
from numpy import *
import pylab as plt

"""
This script is an example workflow script for using the Multiscale Mutation Clustering Algorithm.
The underlying data shown here is from the pan-cancer TCGA data set for the gene PTEN. The tumor types included are: 
ACC, BLCA, BRCA, CESC, COAD, GBM, HNSC, KICH, KIRC, KIRP, LAML, LGG, LIHC, LUAD, LUSC, OV, PRAD, READ, SKCM, STAD, THCA, UCEC, UCS
"""

interval = [0, 403] #0 - Length of PTEN (in Amino Acids)

#Data from TCGA:
#ALL Mutations Include Synonymous mutations, indels, non-synonymous mutations, and stop mutations
PTEN_All_Mutations = [233, 284, 394, 373, 214, 162, 3, 73, 135, 46, 182, 198, 118, 116, 59, 128, 88, 47, 10, 24, 123, 124, 93, 48, 48, 287, 287, 256, 124, 16, 15, 347, 73, 259, 18, 140, 258, 258, 92, 126, 214, 65, 171, 321, 229, 130, 67, 233, 252, 312, 130, 325, 15, 130, 125, 130, 233, 172, 130, 299, 14, 130, 131, 233, 277, 136, 76, 167, 326, 36, 321, 315, 159, 335, 107, 202, 130, 90, 333, 297, 246, 270, 46, 105, 173, 233, 44, 130, 159, 335, 317, 101, 130, 328, 204, 233, 243, 170, 123, 182, 85, 335, 70, 132, 126, 371, 98, 178, 276, 277, 173, 130, 136, 336, 71, 173, 240, 42, 36, 105, 221, 230, 107, 239, 319, 129, 233, 95, 130, 74, 336, 247, 130, 209, 88, 319, 25, 3, 271, 112, 272, 233, 170, 136, 171, 14, 93, 15, 97, 195, 265, 93, 93, 180, 180, 163, 288, 288, 145, 170, 135, 123, 67, 317, 263, 2, 3, 130, 125, 196, 276, 298, 71, 299, 176, 122, 224, 110, 17, 232, 239, 173, 318, 107, 322, 105, 247, 27, 10, 251, 68, 25, 1, 121, 272, 321, 175, 250, 246, 293, 234, 234, 334, 92, 175, 7, 123, 128, 235, 327, 320, 346, 129, 92, 130, 171, 245, 59, 247, 248, 61, 245, 193, 233, 175, 119, 326, 321, 48, 240, 46, 328, 111, 151, 168, 148, 247, 7, 68, 130, 165, 165, 52, 105, 298, 136, 298, 29, 38, 177, 251, 288, 256, 130, 73, 152, 272, 162, 178, 43, 56, 108, 162, 246, 317, 265, 27, 265, 39, 66, 99, 127, 130, 36, 47, 233, 265, 24, 6, 20, 24, 33, 126, 28, 139, 182, 252, 130, 317, 130, 318, 335, 317, 130, 130, 233, 217, 66, 130, 266, 130, 338, 130, 252, 6, 152, 63, 182, 72, 93, 258, 317, 130, 116, 173, 233, 86, 23, 182, 136, 233, 130, 317, 43, 236, 59, 233, 2, 129, 130, 4, 25, 130, 170, 130, 45, 136, 130, 298, 130, 140, 298, 2, 70, 130, 130, 233, 233, 130, 201, 300, 130, 16, 150, 130, 92, 150, 172, 301, 130, 142, 270, 130, 325, 127, 140, 317, 96, 7, 47, 128, 145, 126, 323, 130, 130, 33, 138, 308, 335, 150, 11, 68, 246, 299, 76, 202, 24, 65, 165, 7, 173, 341, 130, 23, 130, 247, 130, 165, 339, 8, 247, 130, 130, 92, 317, 130, 47, 318, 130, 130, 130, 149, 52, 342, 71, 130, 130, 369, 94, 207, 0, 130, 233, 92, 319, 130, 270, 130, 86, 87, 212, 290, 319, 34, 343, 157, 160, 312, 233, 127, 130, 307, 91, 130, 130, 129, 130, 146, 317, 328, 40, 233, 130, 39, 96, 130, 215, 300, 233, 31, 211, 313, 130, 240, 300, 136, 34, 45, 127, 132, 181, 260, 300, 130, 255, 327, 165, 336, 130, 54, 335, 7, 67, 130, 130, 194, 260, 146, 301, 112, 261, 130, 327, 233, 344, 233, 69, 123, 209, 274, 140, 317, 130, 155, 124, 168, 169, 130, 201, 87, 130, 144, 210, 129, 130, 233, 16, 130, 130, 240, 317, 73, 130, 233, 275, 27, 40, 134, 30, 336, 88, 195, 130, 317, 155, 130, 132, 233, 130, 233, 201, 341, 130, 126, 24, 38, 92, 17, 311]
#Just synonymous mutations
PTEN_Synonymous_Mutations = [172, 172, 246, 66, 20, 33, 300, 246, 160, 300, 300]

noise_estimate = 1.0*len(PTEN_Synonymous_Mutations)/len(PTEN_All_Mutations)

#Default bandwidths not used here for faster running time. To try the default settings, set bandwidths = None.
#The below results differ from the published results due to the different set of bandwidths used.
#Before changing the bandwidths manually, please read the documentation as the algorithm works best when dense sets of bandwidths are used.
#All other algorithm parameters use their default values. Details can be found in the documentation and in MultiscaleMutationClusteringScript.py.
bandwidths = [10, 15, 20]
final_clusters, final_params, density_list = M2C.M2C(PTEN_All_Mutations, noise_estimate, bandwidths = bandwidths, interval = interval, print_outs = False)


"""
No Analysis Occurs Below - Results are just plotted using Matplotlib
"""
plt.figure()
color_list = ['blue', 'green', 'cyan', 'yellow', 'orange', 'red', 'purple']
plt.subplot(211)
plt.title("Mutation Histogram\nAll Mutations in gray with colored clusters overlayed on top")
#Create a histogram for all the data
plt.hist(PTEN_All_Mutations, bins = range(interval[1]), color = "gray")
#Create a Histogram for each cluster on top of all the data.
for i in range(len(final_clusters)):
    cluster = final_clusters[i]
    plt.hist(cluster, bins = range(interval[1]), color = color_list[i%len(color_list)], label = str(min(cluster))+"-"+str(max(cluster)))
plt.xlim(interval[0], interval[1])
plt.ylabel("Mutation Count")
plt.legend()

plt.subplot(212)
plt.title("KDE's and Final Mixture Model Gaussians")
x_steps = arange(interval[0], interval[1], .1)
#Plot each KDE
for i in range(len(bandwidths)):
    density = density_list[i]
    h = bandwidths[i]
    plt.plot(x_steps, density, "--", label = "h="+str(h), color = str(.8*i/len(bandwidths)))
#Plot Each Gaussian representing a cluster. Uniform noise is not plotted
for i in range(len(final_params)):
    w, m, s = final_params[i]
    plt.plot(x_steps, [w/(sqrt(2*pi*s**2))*exp(-(x-m)**2/(2*s**2)) for x in x_steps], color_list[i%len(color_list)])
plt.legend()
plt.xlim(interval[0], interval[1])
plt.xlabel("Amino Acid Position")
plt.ylabel("Mutation Density")
plt.show()
