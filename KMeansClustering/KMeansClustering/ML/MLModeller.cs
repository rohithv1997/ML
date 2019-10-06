using KMeansClustering.Model;
using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace KMeansClustering.ML
{
    public class MLModeller
    {
        public static void InitializeMLModeller(MLContext mlContext, IDataView dataView, string rootDirectory)
        {
            var pipeline = mlContext.Transforms
                .Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));

            Console.WriteLine("Model Training Initiated");

            var model = pipeline.Fit(dataView);
            Console.WriteLine("Model Training Completed");
            string _modelPath =GetOutputFile(rootDirectory);

            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }

            Console.WriteLine("Predicting a sample flower.");
            var prediction = mlContext.Model.CreatePredictionEngine<IrisDataModel, ClusterPredictionModel>(model)
                .Predict(
                    new IrisDataModel()
                    {
                        SepalLength = 3.3f,
                        SepalWidth = 1.6f,
                        PetalLength = 0.2f,
                        PetalWidth = 5.1f
                    });



            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");

            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
        }
        private static string GetOutputFile(string rootDirectory) => Directory.GetFiles(rootDirectory, "model.zip", SearchOption.AllDirectories).FirstOrDefault();
    }
}
