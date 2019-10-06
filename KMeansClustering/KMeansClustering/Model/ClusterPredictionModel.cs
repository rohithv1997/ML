using Microsoft.ML.Data;

namespace KMeansClustering.Model
{
    public class ClusterPredictionModel
    {
        [ColumnName("PredictedLabel")] 
        public uint PredictedClusterId { get; set; }

        [ColumnName("Score")] 
        public float[] Distances { get; set; }
    }
}
