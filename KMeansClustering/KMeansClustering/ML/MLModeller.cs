using KMeansClustering.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.IO;

namespace KMeansClustering.ML
{
    public class MLModeller
    {
        public static void InitializeMLModeller(ref MLContext mlContext, ref IDataView dataView)
        {
            var pipeline = mlContext.Transforms
                .Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));
           
            Console.WriteLine("Model Training Initiated"); //Console.ReadLine();
            
            
            
            //Console.WriteLine(dataView.Preview());
           
            var model = pipeline.Fit(dataView);
            Console.WriteLine("Model Training Completed"); //Console.ReadLine();
            string _modelPath=@"G:\Rohith\Coding\ML\KMeansClustering\KMeansClustering\Files\model.zip";

            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
           
            //var predictions=model.Transform(dataView);
            //var metrics=mLContext.Clustering.Evaluate(predictions);
            Console.WriteLine("Predicting a sample flower."); //Console.ReadLine();
            var prediction = mlContext.Model.CreatePredictionEngine<IrisDataModel,ClusterPredictionModel>(model)
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
    }
}
