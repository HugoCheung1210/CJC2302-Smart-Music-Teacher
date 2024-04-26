// sth like a hero component with get started
import React from "react";
import { Link } from "react-router-dom";
import NavbarWithMegaMenu from "./NavbarWithMegaMenu";
import { Button, Typography } from "@material-tailwind/react";
import { MusicalNoteIcon, AcademicCapIcon } from "@heroicons/react/24/solid";


function Homepage() {
  return (
    <div className=" h-screen ">
      <NavbarWithMegaMenu />
      <div className="flex flex-col items-center justify-center h-5/6 ">
        <AcademicCapIcon className="h-20 w-20 text-black my-3" />
        <Typography variant="h1" className="mb-4 text-4xl">
          Welcome to PianoPal
        </Typography>

        <p className="text-lg mb-4">Your personal piano learning assistant</p>
        <Link to="/pieces/1">
          <Button size="lg" >
            Get Started
          </Button>
        </Link>
      </div>
    </div>
  );
}

export default Homepage;
