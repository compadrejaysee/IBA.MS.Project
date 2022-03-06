import * as React from "react";
import { useState, useEffect } from "react";
import { Grid, GridColumn } from "@progress/kendo-react-grid";
// import products from "../products.json";
import db from '../firebase'
import { getDatabase, ref, onValue } from "firebase/database";


const modelStatusRef = ref(db, 'ModelStatus');

const ModelStatus = () => {

    const [modelStatus, SetModelStatus] = useState([])

    const fetchData = async () => {
        onValue(modelStatusRef, (snapshot) => {
            const data = snapshot.val();
            if (data) {
                
                let dataValues = Object.entries(data)
                const model = []
                dataValues.map((modelData)=>{
                   model.unshift(modelData[1])
                })
                SetModelStatus(model)
            }
        });   
      }
      useEffect(() => {
        fetchData()
      }, [modelStatus])

    return (
        <Grid
            style={{
                height: "100px",
                width: "200px"
            }}
            data={modelStatus}
        >
            <GridColumn field="modelType" title="Model Status" />            
        </Grid>
    );
};

export default ModelStatus;