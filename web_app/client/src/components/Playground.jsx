import React from "react";
import { Link } from "react-router-dom";
import { useParams } from "react-router-dom";

import NavbarWithMegaMenu from './NavbarWithMegaMenu';

function Playground() {
  return (
    <div>
        <NavbarWithMegaMenu />
        <h1>Playground</h1>
    </div>
  );
}
export default Playground;
