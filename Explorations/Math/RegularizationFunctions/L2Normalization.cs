using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Math.RegularizationFunctions
{
    public class L2Normalization : IRegularizationFunction
    {
        private double _sizeOfTrainingSet = 0.0;
        private double _regularizationConstant = 0.0;

        public L2Normalization(int sizeOfTrainingSet, double regularizationConstant)
        {
            _sizeOfTrainingSet = sizeOfTrainingSet;
            _regularizationConstant = regularizationConstant;
        }

        public double Compute(double weight)
        {
            return (weight / (_sizeOfTrainingSet * 1.0)) * _regularizationConstant;
        }
    }
}
