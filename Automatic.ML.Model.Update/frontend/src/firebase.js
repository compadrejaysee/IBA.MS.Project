// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getDatabase } from "firebase/database";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyA9d5e05IYuBmbx8Q9CM2yHB7HsksIWxKA",
  authDomain: "autotrainml.firebaseapp.com",
  databaseURL: "https://autotrainml-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "autotrainml",
  storageBucket: "autotrainml.appspot.com",
  messagingSenderId: "132299369200",
  appId: "1:132299369200:web:659b983c93ae77c1227035",
  measurementId: "G-CH33TKNMDB"
};

  
  // Initialize Firebase
  const app = initializeApp(firebaseConfig);
  const db = getDatabase()

  export default db
