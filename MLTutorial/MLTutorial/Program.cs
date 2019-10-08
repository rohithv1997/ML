using System;
using System.IO;
using System.Linq;
using System.Reflection;
using MLTutorialML.Model;

namespace MLTutorial
{
    class Program
    {
        static void Main(string[] args)
        {
            // Add input data
            var input = new ModelInput {SentimentText = GetRandomText()};

            // Load model and predict output of sample data
            ModelOutput result = ConsumeModel.Predict(input);
            Console.WriteLine($"Text: {input.SentimentText}\nIs Toxic: {result.Prediction.ToString()}");
        }

        private static string GetRandomText()
        {
            var dataset = File.ReadLines(GetDataset).ToHashSet();
            return dataset.ElementAt(new Random().Next(0, dataset.Count() + 1)).Split('\t').LastOrDefault();
        }

        private static string GetRootDirectory(string currentDirectory = null)
        {
            currentDirectory ??= Directory.GetCurrentDirectory();
            var dirInfo = new DirectoryInfo(currentDirectory);
            return dirInfo.Name != GetAssemblyName ? GetRootDirectory(Directory.GetParent(currentDirectory).ToString()) : currentDirectory;
        }

        private static string GetAssemblyName => Assembly.GetExecutingAssembly().GetName().Name;

        private static string GetDataset => Directory.GetFiles(GetRootDirectory(), "*.tsv", SearchOption.AllDirectories).FirstOrDefault();

    }
}
