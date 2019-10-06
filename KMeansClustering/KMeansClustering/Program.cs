using KMeansClustering.Model;
using KMeansClustering.ML;
using Microsoft.ML;
using System.IO;
using System;
using System.Linq;

namespace KMeansClustering
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);
            var rootDirectory = GetRootDirectory(Directory.GetCurrentDirectory());
            var currentPath = GetDataset(rootDirectory);
            var trainingData = mlContext.Data.LoadFromTextFile<IrisDataModel>(path: currentPath, separatorChar: ',', hasHeader: false);
            MLModeller.InitializeMLModeller(mlContext, trainingData, rootDirectory);
        }

        private static string GetDataset(string rootDirectory) => Directory.GetFiles(rootDirectory, "iris.csv", SearchOption.AllDirectories).FirstOrDefault();

        private static string GetRootDirectory(string currentDirectory)
        {
            if (new DirectoryInfo(currentDirectory).Name != "KMeansClustering")
            {
                return GetRootDirectory(Directory.GetParent(currentDirectory).ToString());
            }
            return currentDirectory;
        }
    }
}
