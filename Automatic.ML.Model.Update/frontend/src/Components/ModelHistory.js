import * as React from "react";
import { useState, useEffect } from "react";
import { Grid, GridColumn } from "@progress/kendo-react-grid";
// import products from "../products.json";
import db from '../firebase'
import { getDatabase, ref, onValue } from "firebase/database";


const modelHistoryRef = ref(db, 'ModelHistory');

const ModelHistory = () => {

    const [modelHistory, SetModelHistory] = useState([])

    const fetchData = async () => {
        onValue(modelHistoryRef, (snapshot) => {
            const data = snapshot.val();
            if (data) {
                
                let dataValues = Object.entries(data)
                const model = []
                dataValues.map((modelData)=>{
                   model.unshift(modelData[1])
                })
                SetModelHistory(model)
            }
        });   
      }
      useEffect(() => {
        fetchData()
      }, [modelHistory])

    return (
        <Grid
            style={{
                height: "400px",
            }}
            data={modelHistory}
        >
            <GridColumn field="TimeStamp" title="Timestamp" width="250px" />
            <GridColumn field="modelType" title="Model Type" />
            <GridColumn field="accurracy" title="Accuracy" />
            
        </Grid>
    );
};

export default ModelHistory;