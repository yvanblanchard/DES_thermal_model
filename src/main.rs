#![allow(warnings)]
mod gcode_reader;
mod interpolator;
mod model;
mod model_generator;
mod model_iso_td_shc_td_con;
mod model_orthotropic_td_shc;
mod model_updater;
mod primitives;
mod simpleModel;
mod model_iso_td_shc_td;
mod model_orthotropic_td_shc_variable_h;
//mod model_with_structural;

use num_cpus;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::borrow::BorrowMut;
use std::cmp::{max, Ordering};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader, BufWriter, Read};
use std::iter::FromIterator;
use std::ops::Index;
use std::thread::current;
use std::{fmt, fs};

use crate::gcode_reader::GCodeReader;
use crate::interpolator::*;
use crate::model::Model;
use crate::model_generator::ModelGenerator;
use crate::model_updater::ModelUpdater;
use crate::primitives::{Node, Point};
use std::str::FromStr;

use std::time::Instant;

use hdf5::File as H5File;
use ndarray::{Array1, Array2};

fn read_next_item<'a>(inputfilebufreader: &'a mut BufReader<File>, line: &'a mut String) -> Option<&'a str>{
    line.clear();
    inputfilebufreader.read_line(line);
    let trimmed = line.trim_end();
    let mut linesplit = trimmed.split(" ");
    linesplit.next();
    return linesplit.next()
}

/// Writes all simulation results to a single HDF5 file.
///
/// HDF5 layout:
///   /mesh/nodes          [N, 3] f64   — (x, y, z) per node
///   /mesh/elements       [E, 8] u64   — 8 node indices (0-based) per element
///   /activation/times         [E] f64
///   /activation/element_ids   [E] u64
///   /activation/layer_nos     [E] u64
///   /activation/orientations  [E, 2] f64
///   /results/elem_time_temp_data    [flat] f64  — alternating (time, temp) pairs
///   /results/elem_time_temp_offsets [E+1]  u64  — CSR row pointers
///   /results/node_time_temp_data    [flat] f64
///   /results/node_time_temp_offsets [N+1]  u64
fn write_hdf5_output(
    filename: &str,
    nodes: &Vec<[f64; 3]>,
    elements: &Vec<[usize; 8]>,
    activation_times: &Vec<(f64, usize, [f64; 2], usize)>,
    elem_temps: &Vec<Vec<f64>>,
    node_temps: &Vec<Vec<f64>>,
) {
    if let Some(parent) = std::path::Path::new(filename).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("can't create HDF5 output directory");
        }
    }

    let h5 = H5File::create(filename).expect("can't create HDF5 output file");

    // ---- /mesh ----
    let mesh_grp = h5.create_group("mesh").expect("can't create mesh group");

    let n_nodes = nodes.len();
    {
        let nodes_flat: Vec<f64> = nodes.iter().flat_map(|n| n.iter().copied()).collect();
        let arr = if n_nodes > 0 {
            Array2::from_shape_vec((n_nodes, 3), nodes_flat).expect("bad node array shape")
        } else {
            Array2::zeros((0, 3))
        };
        mesh_grp.new_dataset_builder().with_data(&arr.view()).create("nodes")
            .expect("can't write mesh/nodes");
    }

    let n_elems = elements.len();
    {
        let elems_flat: Vec<u64> = elements.iter()
            .flat_map(|e| e.iter().map(|&x| x as u64))
            .collect();
        let arr = if n_elems > 0 {
            Array2::from_shape_vec((n_elems, 8), elems_flat).expect("bad element array shape")
        } else {
            Array2::zeros((0usize, 8usize))
        };
        mesh_grp.new_dataset_builder().with_data(&arr.view()).create("elements")
            .expect("can't write mesh/elements");
    }

    // ---- /activation ----
    let act_grp = h5.create_group("activation").expect("can't create activation group");

    let n_act = activation_times.len();
    {
        let times: Vec<f64>  = activation_times.iter().map(|x| x.0).collect();
        let eids:  Vec<u64>  = activation_times.iter().map(|x| x.1 as u64).collect();
        let lnos:  Vec<u64>  = activation_times.iter().map(|x| x.3 as u64).collect();

        act_grp.new_dataset_builder()
            .with_data(&Array1::from_vec(times).view()).create("times")
            .expect("can't write activation/times");
        act_grp.new_dataset_builder()
            .with_data(&Array1::from_vec(eids).view()).create("element_ids")
            .expect("can't write activation/element_ids");
        act_grp.new_dataset_builder()
            .with_data(&Array1::from_vec(lnos).view()).create("layer_nos")
            .expect("can't write activation/layer_nos");

        let ori_flat: Vec<f64> = activation_times.iter()
            .flat_map(|x| x.2.iter().copied())
            .collect();
        let ori_arr = if n_act > 0 {
            Array2::from_shape_vec((n_act, 2), ori_flat).expect("bad orientation array shape")
        } else {
            Array2::zeros((0usize, 2usize))
        };
        act_grp.new_dataset_builder()
            .with_data(&ori_arr.view()).create("orientations")
            .expect("can't write activation/orientations");
    }

    // ---- /results ----
    let res_grp = h5.create_group("results").expect("can't create results group");

    // Elemental temperatures — CSR format
    {
        let mut data: Vec<f64> = Vec::new();
        let mut offsets: Vec<u64> = Vec::with_capacity(elem_temps.len() + 1);
        offsets.push(0);
        for v in elem_temps {
            data.extend_from_slice(v.as_slice());
            offsets.push(data.len() as u64);
        }
        res_grp.new_dataset_builder()
            .with_data(&Array1::from_vec(data).view()).create("elem_time_temp_data")
            .expect("can't write results/elem_time_temp_data");
        res_grp.new_dataset_builder()
            .with_data(&Array1::from_vec(offsets).view()).create("elem_time_temp_offsets")
            .expect("can't write results/elem_time_temp_offsets");
    }

    // Nodal temperatures — CSR format
    {
        let mut data: Vec<f64> = Vec::new();
        let mut offsets: Vec<u64> = Vec::with_capacity(node_temps.len() + 1);
        offsets.push(0);
        for v in node_temps {
            data.extend_from_slice(v.as_slice());
            offsets.push(data.len() as u64);
        }
        res_grp.new_dataset_builder()
            .with_data(&Array1::from_vec(data).view()).create("node_time_temp_data")
            .expect("can't write results/node_time_temp_data");
        res_grp.new_dataset_builder()
            .with_data(&Array1::from_vec(offsets).view()).create("node_time_temp_offsets")
            .expect("can't write results/node_time_temp_offsets");
    }

    println!("HDF5 output written to {}", filename);
}


fn filter_activation_times_by_layer_range(
    activation_times: Vec<(f64, usize, [f64; 2], usize)>,
    start_layer: usize,
    end_layer: Option<usize>,
) -> Vec<(f64, usize, [f64; 2], usize)> {
    if start_layer == 0 && end_layer.is_none() {
        return activation_times;
    }
    let filtered: Vec<_> = activation_times
        .into_iter()
        .filter(|(_, _, _, layer_no)| {
            *layer_no >= start_layer && end_layer.map_or(true, |end| *layer_no <= end)
        })
        .collect();
    if filtered.is_empty() {
        return filtered;
    }
    let t0 = filtered[0].0;
    filtered
        .into_iter()
        .map(|(t, cell, orient, layer)| (t - t0, cell, orient, layer))
        .collect()
}

fn main() {
    let inputfile = File::open("inputfiles/Input_file.txt").expect("InputFile not found");
    let mut inputfilebufreader = BufReader::with_capacity(10000, inputfile);
    let mut line = String::with_capacity(100);
    inputfilebufreader.read_line(&mut line).expect("cant read line 1");
    line.clear();
    inputfilebufreader.read_line(&mut line).expect("cant read line 2");
    let mut linesplit = line.split(" "); linesplit.next();
    let mut num_cpus = usize::from_str(linesplit.next().expect("cant read number of cpus").trim()).expect("cant parse number of cpus provided to integer");
    if num_cpus == 0 {
        num_cpus = num_cpus::get_physical();
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus+2)
        .build()
        .expect("can't create threads");
    println!("using {} cpu threads", num_cpus);
    let maxthreads = num_cpus*2;
    line.clear();
    inputfilebufreader.read_line(&mut line).expect("cant read line 3");
    let mut linesplit = line.split(" "); linesplit.next();
    let gcodefile = linesplit.next().expect("cant read the input gcode filename").trim();
    println!("reading gcode file |{}|", gcodefile);
    let gcr = GCodeReader::new(gcodefile);
    line.clear();
    inputfilebufreader.read_line(&mut line).expect("cant read line 3");
    let mut linesplit = line.split(" "); linesplit.next();
    let divs_per_bead = usize::from_str(linesplit.next().expect("cant read divisions per beadwidth").trim()).expect("cant read divisions per beadwidth as positive integer");
    line.clear();
    inputfilebufreader.read_line(&mut line).expect("cant read line 3");
    let mut linesplit = line.split(" "); linesplit.next();
    let divs_per_bead_z = usize::from_str(linesplit.next().expect("cant read divisions per beadheight").trim()).expect("cant read divisions per beadheight as positive integer");
    let beadwidth = f64::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read beadwidth")).expect("cant parse beadwidth to float");
    let beadheight = f64::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read beadheight")).expect("cant parse beadheight to float");

    let model_type = usize::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read model type")).expect("cant parse model type to integer");
    if model_type == 0 {

    }
    else if model_type == 1 {
        let sp_ht_cap_filename = String::from(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read specific heat capacity filename").trim());
        let sp_heat_cap_step = f64::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read specific heat capacity file temperature spacing").trim()).expect("cant parse specific heat capacity file temperature spacing to float");
        let conductivity_filename = String::from(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read conductivity filename").trim());
        let conductivity_step = f64::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read conductivity file temperature spacing").trim()).expect("cant parse conductivity file temperature spacing to float");
        let density = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read density").trim()).expect("cant parse density to float");
        let h = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read convective heat film transfer coefficient").trim()).expect("cant parse convective heat transfer film coefficient to float");
        let temp_bed = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read bed temperature").trim()).expect("cant parse bed temperature to float");
        let bed_k = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read bed interface conductivity").trim()).expect("cant parse bed interface conductivity to float");
        let ambient_temp = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read ambient temperature").trim()).expect("cant parse ambient temperature to float");

        let extrusion_temperature = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read extrusion temperature").trim()).expect("cant parse extrusion temperature to float");
        let emissivity = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read emissivity").trim()).expect("cant parse emissivity to float");
        let time_step = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read time step").trim()).expect("cant parse time step to float");
        let cooldown_period = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read cooldown period").trim()).expect("cant parse cooldown period to float");

        let hdf5_output_name = String::from(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read hdf5 output filename").trim());
        let min_temp_change_store = f64::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read min temp change to store").trim()).expect("cant parse min temp change to store to float");
        let turn_off_layers_at = usize::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read turn off layer at").trim()).expect("cant parse turn off layers at as integer");
        let start_layer = usize::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read start layer").trim()).expect("cant parse start layer as integer");
        let end_layer_raw = isize::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read end layer").trim()).expect("cant parse end layer as integer");
        let end_layer: Option<usize> = if end_layer_raw < 0 { None } else { Some(end_layer_raw as usize) };

        // all inputs done. calculations start here
        let element_width = beadwidth / divs_per_bead as f64;

        let element_height = beadheight / divs_per_bead_z as f64;
        let xdiv = ((gcr.xmax - gcr.xmin + beadwidth) / element_width + 1.0).round() as usize;
        let ydiv = ((gcr.ymax - gcr.ymin + beadwidth) / element_width + 1.0).round() as usize;
        let zdiv = ((gcr.zmax - gcr.zmin + beadheight) / element_height + 1.0).round() as usize;

        let zmin = gcr.zmin.clone();
        println!(
            "xmin {} xmax {} ymin {} ymax {} zmin {} zmax {} xdiv {} ydiv {} zdiv {}",
            gcr.xmin, gcr.xmax, gcr.ymin, gcr.ymax, gcr.zmin, gcr.zmax, xdiv, ydiv, zdiv
        );
        let init_temp = extrusion_temperature;
        let tic = Instant::now();
        let mut m = ModelGenerator::new(
            gcr.xmin - beadwidth / 2.0,
            gcr.xmax + beadwidth / 2.0,
            gcr.ymin - beadwidth / 2.0,
            gcr.ymax + beadwidth / 2.0,
            gcr.zmin - beadheight,
            gcr.zmax,
            xdiv,
            ydiv,
            zdiv,
            init_temp,
            element_width,
            divs_per_bead_z,
        );
        println!("time taken to generate model {:?}", tic.elapsed());

        let mut activation_times_all: Vec<(f64,usize,[f64;2],usize)> = m.generate_activation_times_all_layers(
            gcr.segment,
            gcr.is_extrusion_on,
            gcr.speed,
            beadwidth,
            beadheight,
            &pool,
            maxthreads,
        );

        activation_times_all = filter_activation_times_by_layer_range(activation_times_all, start_layer, end_layer);
        println!("simulating layer range {} to {}, {} elements activated", start_layer,
            end_layer.map_or_else(|| "last".to_string(), |e| e.to_string()), activation_times_all.len());
        println!("writing node and element files");
        let mut activated_elements = Vec::with_capacity(activation_times_all.len());
        for i in 0..activation_times_all.len(){
            activated_elements.push(activation_times_all[i].1);
        }
        let (nodeveclen, elemveclen, nodesnew, elementsupdated, nodenum_old_to_new) =
            m.get_nodes_and_elements(activated_elements.clone());

        let sp_heat_cap_interpolation_table = Interpolator::read_data_from_file(sp_ht_cap_filename.as_str(), sp_heat_cap_step, maxthreads);
        let conductivity_interpolation_table = Interpolator::read_data_from_file(conductivity_filename.as_str(), conductivity_step, maxthreads);

        println!("specific heat capacity and conductivity data files read");
        let mut mu = ModelUpdater::new(activation_times_all.clone(), m, nodenum_old_to_new);

        let input_data = ([0.205, 0.205, 0.205], density, 1500.0, h, [init_temp, ambient_temp]);

        let mut mdl = model_iso_td_shc_td::Model::new(
            nodesnew.clone(),
            mu.activation_times.len(),
            sp_heat_cap_interpolation_table,
            element_width,
            element_height,
            turn_off_layers_at,
            zmin - beadheight, input_data.1, emissivity, bed_k, temp_bed, maxthreads);

        let areas_and_dists = [
            element_width * element_height,
            element_width * element_width,
            element_height * element_width / element_width,
            element_width * element_width / element_height,
            element_width * element_width * element_height,
        ];
        let mut tmpfile = File::create("tempfile.csv").expect("cant create temporary scratch file");
        let mut bw = BufWriter::with_capacity(10000, tmpfile);

        let mut datastorer = Vec::with_capacity(activated_elements.len());
        for _ in 0..activated_elements.len(){
            datastorer.push(Vec::with_capacity(1000));
        }

        let mut datastorernode: Vec<Vec<f64>> = Vec::new();

        let mut nd_to_elem = Vec::with_capacity(nodeveclen);
        for _ in 0..nodeveclen{
            nd_to_elem.push(Vec::with_capacity(6));
        }
        for i in 0..elementsupdated.len(){
            for j in 0..8{
                nd_to_elem[elementsupdated[i][j]].push(i);
            }
        }

        println!("starting simulation timestep {}", time_step);
        let tic = Instant::now();
        model_iso_td_shc_td::Model::update_model(
            &mut mu,
            &mut mdl,
            &mut bw,
            areas_and_dists,
            time_step,
            &conductivity_interpolation_table,
            maxthreads,
            input_data,
            min_temp_change_store,
            &mut datastorer,
            &mut datastorernode,
            nd_to_elem.as_slice(),
            cooldown_period,
            &pool,
        );
        println!("simulation ended in {:?}. writing HDF5 output...", tic.elapsed());
        let tic = Instant::now();
        write_hdf5_output(
            &hdf5_output_name,
            &nodesnew,
            &elementsupdated,
            &activation_times_all,
            &datastorer,
            &datastorernode,
        );
        println!("HDF5 output written in {:?}", tic.elapsed());
    }

    else if model_type == 2 {
        let sp_ht_cap_filename = String::from(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read specific heat capacity filename").trim());
        let sp_heat_cap_step = f64::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read specific heat capacity file temperature spacing").trim()).expect("cant parse specific heat capacity file temperature spacing to float");
        let kx = f64::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read kx").trim()).expect("cant parse kx to float");
        let ky = f64::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read ky").trim()).expect("cant parse ky to float");
        let kz = f64::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read kz").trim()).expect("cant parse kz to float");

        println!("kx ky kz {},{},{}", kx, ky, kz);

        let density = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read density").trim()).expect("cant parse density to float");
        let h = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read convective heat film transfer coefficient").trim()).expect("cant parse convective heat transfer film coefficient to float");
        let temp_bed = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read bed temperature").trim()).expect("cant parse bed temperature to float");
        let bed_k = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read bed interface conductivity").trim()).expect("cant parse bed interface conductivity to float");
        let ambient_temp = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read ambient temperature").trim()).expect("cant parse ambient temperature to float");

        let extrusion_temperature = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read extrusion temperature").trim()).expect("cant parse extrusion temperature to float");
        let emissivity = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read emissivity").trim()).expect("cant parse emissivity to float");
        let time_step = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read time step").trim()).expect("cant parse time step to float");
        let cooldown_period = f64::from_str(read_next_item(&mut inputfilebufreader,&mut line).expect("cant read cooldown period").trim()).expect("cant parse cooldown period to float");

        let hdf5_output_name = String::from(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read hdf5 output filename").trim());
        let min_temp_change_store = f64::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read min temp change to store").trim()).expect("cant parse min temp change to store to float");
        let turn_off_layers_at = usize::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read turn off layer at").trim()).expect("cant parse turn off layers at as integer");
        let start_layer = usize::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read start layer").trim()).expect("cant parse start layer as integer");
        let end_layer_raw = isize::from_str(read_next_item(&mut inputfilebufreader, &mut line).expect("cant read end layer").trim()).expect("cant parse end layer as integer");
        let end_layer: Option<usize> = if end_layer_raw < 0 { None } else { Some(end_layer_raw as usize) };

        // all inputs done. calculations start here
        let element_width = beadwidth / divs_per_bead as f64;

        let element_height = beadheight / divs_per_bead_z as f64;
        let xdiv = ((gcr.xmax - gcr.xmin + beadwidth) / element_width + 1.0).round() as usize;
        let ydiv = ((gcr.ymax - gcr.ymin + beadwidth) / element_width + 1.0).round() as usize;
        let zdiv = ((gcr.zmax - gcr.zmin + beadheight) / element_height + 1.0).round() as usize;

        let zmin = gcr.zmin.clone();
        println!(
            "xmin {} xmax {} ymin {} ymax {} zmin {} zmax {} xdiv {} ydiv {} zdiv {}",
            gcr.xmin, gcr.xmax, gcr.ymin, gcr.ymax, gcr.zmin, gcr.zmax, xdiv, ydiv, zdiv
        );
        let init_temp = extrusion_temperature;
        let tic = Instant::now();
        let mut m = ModelGenerator::new(
            gcr.xmin - beadwidth / 2.0,
            gcr.xmax + beadwidth / 2.0,
            gcr.ymin - beadwidth / 2.0,
            gcr.ymax + beadwidth / 2.0,
            gcr.zmin - beadheight,
            gcr.zmax,
            xdiv,
            ydiv,
            zdiv,
            init_temp,
            element_width,
            divs_per_bead_z,
        );
        println!("time taken to generate model {:?}", tic.elapsed());

        let mut activation_times_all: Vec<(f64,usize,[f64;2],usize)> = m.generate_activation_times_all_layers(
            gcr.segment,
            gcr.is_extrusion_on,
            gcr.speed,
            beadwidth,
            beadheight,
            &pool,
            maxthreads,
        );

        activation_times_all = filter_activation_times_by_layer_range(activation_times_all, start_layer, end_layer);
        println!("simulating layer range {} to {}, {} elements activated", start_layer,
            end_layer.map_or_else(|| "last".to_string(), |e| e.to_string()), activation_times_all.len());
        println!("writing node and element files");
        let mut activated_elements = Vec::with_capacity(activation_times_all.len());
        for i in 0..activation_times_all.len(){
            activated_elements.push(activation_times_all[i].1);
        }
        let (nodeveclen, elemveclen, nodesnew, elementsupdated, nodenum_old_to_new) =
            m.get_nodes_and_elements(activated_elements.clone());

        let sp_heat_cap_interpolation_table = Interpolator::read_data_from_file(sp_ht_cap_filename.as_str(), sp_heat_cap_step, maxthreads);

        println!("specific heat capacity data file read");
        let mut mu = ModelUpdater::new(activation_times_all.clone(), m, nodenum_old_to_new);

        let input_data = ([kx, ky, kz], density, 1500.0, h, [init_temp, ambient_temp]);

        let mut mdl = model_orthotropic_td_shc_variable_h::Model::new(
            nodesnew.clone(),
            mu.activation_times.len(),
            sp_heat_cap_interpolation_table,
            element_width,
            element_height,
            turn_off_layers_at,
            zmin - beadheight, input_data.1, emissivity, bed_k, temp_bed, maxthreads);

        let areas_and_dists = [
            element_width * element_height,
            element_width * element_width,
            element_height * element_width / element_width,
            element_width * element_width / element_height,
            element_width * element_width * element_height,
        ];
        let mut tmpfile = File::create("tempfile.csv").expect("cant create temporary scratch file");
        let mut bw = BufWriter::with_capacity(10000, tmpfile);

        let mut datastorer = Vec::with_capacity(activated_elements.len());
        for _ in 0..activated_elements.len(){
            datastorer.push(Vec::with_capacity(1000));
        }

        let mut datastorernode: Vec<Vec<f64>> = Vec::new();

        let mut nd_to_elem = Vec::with_capacity(nodeveclen);
        for _ in 0..nodeveclen{
            nd_to_elem.push(Vec::with_capacity(6));
        }
        for i in 0..elementsupdated.len(){
            for j in 0..8{
                nd_to_elem[elementsupdated[i][j]].push(i);
            }
        }

        println!("starting simulation timestep {}", time_step);
        let tic = Instant::now();
        model_orthotropic_td_shc_variable_h::Model::update_model(
            &mut mu,
            &mut mdl,
            &mut bw,
            areas_and_dists,
            time_step,
            maxthreads,
            input_data,
            min_temp_change_store,
            &mut datastorer,
            &mut datastorernode,
            nd_to_elem.as_slice(),
            cooldown_period,
            &pool,
        );
        println!("simulation ended in {:?}. writing HDF5 output...", tic.elapsed());
        let tic = Instant::now();
        write_hdf5_output(
            &hdf5_output_name,
            &nodesnew,
            &elementsupdated,
            &activation_times_all,
            &datastorer,
            &datastorernode,
        );
        println!("HDF5 output written in {:?}", tic.elapsed());
    }
}
