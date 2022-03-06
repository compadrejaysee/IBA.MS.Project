import * as React from "react";
import { useState, useEffect } from "react";
import { Grid, GridColumn } from "@progress/kendo-react-grid";
// import products from "../products.json";
import db from '../firebase'
import { getDatabase, ref, onValue } from "firebase/database";


const modelAccRef = ref(db, 'ModelAcc');

const ModelAcc = () => {

    const [modelAcc, SetModelAcc] = useState([])

    const fetchData = async () => {
        onValue(modelAccRef, (snapshot) => {
            const data = snapshot.val();
            if (data) {
                
                let dataValues = Object.entries(data)
                const model = []
                dataValues.map((modelData)=>{
                   model.unshift(modelData[1])
                })
                SetModelAcc(model)
            }
        });   
      }
      useEffect(() => {
        fetchData()
      }, [modelAcc])

    return (
        <Grid
        style={{
            height: "100px",
            width: "200px"
        }}
        data={modelAcc}
    >
        <GridColumn field="accurracy" title="Accuracy" />
        
    </Grid>
    );
};

export default ModelAcc;