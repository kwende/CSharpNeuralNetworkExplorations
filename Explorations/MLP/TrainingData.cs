using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public class TrainingData
    {
        public double[] X { get; set; }
        public double[] Y { get; set; }

        public int XWidth { get; set; }
        public int XHeight { get; set; }
        public int Label { get; set; }

        public override string ToString()
        {
            return string.Join(",", X) + " = " + string.Join(",", Y);
        }
    }
}
