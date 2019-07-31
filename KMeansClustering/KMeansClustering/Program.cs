using KMeansClustering.Model;
using KMeansClustering.ML;
using Microsoft.ML;
using System;
using System.IO;

namespace KMeansClustering
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed:0);
            var currentPath = @"G:\Rohith\Coding\ML\KMeansClustering\KMeansClustering\DataSet\iris.csv";
            var trainingData = mlContext.Data.LoadFromTextFile<IrisDataModel>(path: currentPath, separatorChar: ',', hasHeader: false);
            MLModeller.InitializeMLModeller(ref mlContext, ref trainingData);
        }
    }
}
