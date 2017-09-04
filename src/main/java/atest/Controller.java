package atest;

import atest.data.MongoData;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.context.request.WebRequest;

import java.util.HashMap;
import java.util.Map;

public class Controller {

    final Gson gson = new GsonBuilder().setPrettyPrinting().create();

    private String messageOK() {
        HashMap<String, String> ret = new HashMap<String, String>();
        ret.put("message", "OK");
        return gson.toJson(ret);
    }

    @RequestMapping( value = "/ping", method = RequestMethod.GET)
    public @ResponseBody String ping () {
        return messageOK();
    }

    //put insert
    @RequestMapping(value = "/add", method = RequestMethod.POST)
    public @ResponseBody String add(@RequestBody String requestBody){
        return MongoData.get_instance().insert(requestBody);
    }

    //PUT update
    @RequestMapping(value= "/update", method = RequestMethod.POST)
    public @ResponseBody String update(@RequestBody String requestBody) {
        return MongoData.get_instance().update(requestBody);
    }


    //DELETE delete
    @RequestMapping(value= "/delete/{id}", method = RequestMethod.DELETE)
    public @ResponseBody String delete(@PathVariable String id){
        MongoData.get_instance().delete(id);
        return messageOK();
    }
    //GET get by id
    @RequestMapping(value= "/id/{id}", method = RequestMethod.GET)
    public @ResponseBody String id(@PathVariable String id){
        return MongoData.get_instance().findById(id);
    }
    //GET search ( accept regex )
    @RequestMapping(value= "/search", method = RequestMethod.GET)
    public @ResponseBody String search(WebRequest webRequest){
        Map<String, String[]> params = webRequest.getParameterMap();
        int size = params.size();
        String[] parameters = new String[size];
        int count = 0;
        for(String key: params.keySet()){
            String[] value = params.get(key);
            parameters[count] = key+"="+value[0];
            count ++;
        }
        return MongoData.get_instance().find(parameters);
    }
}

