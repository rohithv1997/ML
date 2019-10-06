using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLTutorialML.ConsoleApp;

namespace MLTest
{
    [TestClass]
    public class MLTest
    {
        [TestMethod]
        public void TrainModel()
        {
            ModelBuilder.CreateModel();
        }
    }
}
