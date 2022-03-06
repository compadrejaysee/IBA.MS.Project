import * as React from "react";
import { useState, useEffect } from "react";
import { Grid, GridColumn } from "@progress/kendo-react-grid";
// import products from "../products.json";
import db from '../firebase'
import { getDatabase, ref, onValue } from "firebase/database";


const currentModelRef = ref(db, 'CurrentModel');

const CurrentModel = () => {

    const [currentModel, SetCurrentModel] = useState([])

    const fetchData = async () => {
        onValue(currentModelRef, (snapshot) => {
            const data = snapshot.val();
            if (data) {
                
                let dataValues = Object.entries(data)
                const model = []
                dataValues.map((modelData)=>{
                   model.unshift(modelData[1])
                })
                SetCurrentModel(model)
            }
        });   
      }
      useEffect(() => {
        fetchData()
      }, [currentModel])

    return (
        <Grid
            style={{
                height: "100px",
            }}
            data={currentModel}
        >
            <GridColumn field="modelType" title="Model Type" />
            <GridColumn field="accurracy" title="Model Accuracy" />
            
        </Grid>
    );
};

export default CurrentModel;