using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Interfaces
{
    public interface IRegularizationFunction
    {
        double Compute(double weight, int sizeOfTrainingSet);
    }
}
