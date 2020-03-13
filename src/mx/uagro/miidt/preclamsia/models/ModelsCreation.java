/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mx.uagro.miidt.preclamsia.models;

import weka.classifiers.trees.RandomForest;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.SerializationHelper;

/**
 *
 * @author RabDos
 */
public class ModelsCreation {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/"
                + "assets/Preeclampsia.arff");
        Instances trainData = source.getDataSet();
        trainData.setClassIndex(trainData.numAttributes()-1);
        RandomForest rForest = new RandomForest();
        IBk iBk = new IBk();
        J48 j48 = new J48();
        rForest.buildClassifier(trainData);
        iBk.buildClassifier(trainData);
        j48.buildClassifier(trainData);
        System.out.println("\n" + rForest);
        System.out.println("\n" + iBk);
        System.out.println("\n" + j48);
        
        SerializationHelper.write("src/assets/RandomForest.model", rForest);
        SerializationHelper.write("src/assets/IBk.model", iBk);
        SerializationHelper.write("src/assets/J48.model", j48);
        
        System.out.println("Models saved!!!");
    }
}
