using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public class ValidationDataOptions
    {
        public List<TrainingData> ValidationData { get; set; }
        public int NumberOfEpochsBetweenTests { get; set; }
    }
}
