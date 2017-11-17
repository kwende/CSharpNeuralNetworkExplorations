using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.DataStructures
{
    public class LearningProgress
    {
        public int Counter { get; set; }
        public int BatchNumber { get; set; }
        public int Epoch { get; set; }
        public double CurrentNetworkError { get; set; }
    }
}
