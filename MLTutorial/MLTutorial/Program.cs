using System;
using System.Diagnostics;
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
            var input = new ModelInput();
            input.SentimentText = GetRandomText();

            // Load model and predict output of sample data
            ModelOutput result = ConsumeModel.Predict(input);
            Console.WriteLine($"Text: {input.SentimentText}\nIs Toxic: {result.Prediction}");
        }

        private static string GetRandomText()
        {
            var dataset = File.ReadLines(GetDataset(GetRootDirectory()));
            return dataset.ElementAt(new Random().Next(0, dataset.Count() + 1)).Split('\t').LastOrDefault();
        }

        private static string GetRootDirectory(string currentDirectory = null)
        {
            currentDirectory ??= Directory.GetCurrentDirectory();
            if (new DirectoryInfo(currentDirectory).Name != GetAssemblyName)
            {
                return GetRootDirectory(Directory.GetParent(currentDirectory).ToString());
            }
            return currentDirectory;
        }

        private static string GetAssemblyName => Assembly.GetExecutingAssembly().GetName().Name;

        private static string GetDataset(string rootDirectory) => Directory.GetFiles(rootDirectory, "*.tsv", SearchOption.AllDirectories).FirstOrDefault();

    }
}
