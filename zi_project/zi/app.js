import express       from 'express';
import bodyParser    from 'body-parser';
import path          from 'path';
import {PythonShell} from 'python-shell';
import fs            from 'fs';

const app = express();

app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(bodyParser.urlencoded({
  extended: true
}));
app.use(bodyParser.json());

const router = express.Router();

router.get('/login', (req, res) => {
    res.render('login', { });
});

router.post('/login', (req, res) => {
    res.redirect('/input');
});


router.get('/input', (req, res) => {
    res.render('input', { });
});

router.post('/send', (req, res) => {
    const name = req.body.name;
    const objLetters = {};
    objLetters.letters = req.body.letters
    const letters = JSON.stringify(objLetters);
    fs.writeFile(`./../${name}.json`, letters, function(err, result) {
        if(err) console.log('error', err);
    });

    PythonShell.run('create_image.py', null, function (err, results) {
      if (err) console.log(err);
    });

    PythonShell.run('neural.py', null, function (err, results) {
      if (err) console.log(err);
    });
    res.redirect('/input');
})

app.use('/', router);

app.use((req, res, next) => {
  const err = new Error('Not Found');
  err.status = 404;
  next(err);
});

app.use(  (err, req, res, next) => {
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  res.status(err.status || 500);
  res.render('error');
});

app.listen(3001, () => {
    console.log(`APP STARTING AT PORT 3001`);
});
